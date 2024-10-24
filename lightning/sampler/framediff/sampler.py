import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import pandas as pd
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional

from lightning.sampler.framediff import utils as su
from preprocess.tools import utils as du
from preprocess.tools import residue_constants
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from evaluate.openfold.data import data_transforms
import esm
import copy
from lightning.model.framediff.lightning_model import framediff_Lightning_Model
from evaluate.openfold.utils import rigid_utils as ru
from preprocess.tools import all_atom


class framediff_Sampler:
    def __init__(self, conf: DictConfig):

        self.conf = conf
        self.infer_conf = conf.sampler.inference
        self.diff_conf = self.infer_conf.diffusion
        self.sample_conf = self.infer_conf.samples
        self.data_conf = self.conf.dataset
        self.model_conf = self.conf.model
        self.log = logging.getLogger(__name__)
        self.output_dir = self.infer_conf.output_dir

        # Set-up accelerator
        if torch.cuda.is_available():
            if self.infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self.infer_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self.log.info(f'Using device: {self.device}')

        # Load model from checkpoint
        self._rng = np.random.default_rng(self.infer_conf.seed)
        self.inference_ckpt = self.infer_conf.weights_path
        self.lightning_model = framediff_Lightning_Model.load_from_checkpoint(self.inference_ckpt)

    def set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.lightning_model.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        if num_t is None:
            num_t = self.data_conf.num_t
        if min_t is None:
            min_t = self.data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1 / num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self.model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self.set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder)
                sample_feats = self.lightning_model.self_conditioning(sample_feats)
            for t in reverse_steps:
                if t > min_t:
                    sample_feats = self.set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.lightning_model.model(sample_feats)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    if self.model_conf.embed.embed_self_conditioning:
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    rigids_t = self.lightning_model.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=center,
                        noise_scale=noise_scale,
                    )
                else:
                    model_out = self.lightning_model.model(sample_feats)
                    rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                psi_pred = model_out['psi']
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred),
                        psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(
                    rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            'prot_traj': all_bb_prots,
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['psi_pred'] = psi_pred[None]
            ret['rigid_0_traj'] = all_bb_0_pred
        return ret

    def sample(self, sample_length: int):
        """Sample based on length.
        Args:
            sample_length: length to sample

        Returns:
            Sample outputs. See self.inference_fn.
        """
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        # Initialize data
        ref_sample = self.lightning_model.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )

        res_idx = torch.arange(1, sample_length+1)
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2)),
            'sc_ca_t': np.zeros((sample_length, 3)),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)

        # Run inference
        sample_out = self.inference_fn(
            init_feats,
            num_t=self.diff_conf.num_t,
            min_t=self.diff_conf.min_t,
            aux_traj=True,
            noise_scale=self.diff_conf.noise_scale,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)

    def save_traj(
            self,
            bb_prot_traj: np.ndarray,
            x0_traj: np.ndarray,
            diffuse_mask: np.ndarray,
            output_dir: str
    ):
        """Writes final sample and reverse diffusion trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file os all intermediate diffused states.
                'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for diffused residues and 0 for motif
            residues if there are any.
        """

        # Write sample.
        diffuse_mask = diffuse_mask.astype(bool)
        sample_path = os.path.join(output_dir, 'sample')
        prot_traj_path = os.path.join(output_dir, 'bb_traj')
        x0_traj_path = os.path.join(output_dir, 'x0_traj')

        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        sample_path = su.write_prot_to_pdb(
            bb_prot_traj[0],
            sample_path,
            b_factors=b_factors
        )
        prot_traj_path = su.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors
        )
        x0_traj_path = su.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors
        )
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            'x0_traj_path': x0_traj_path,
        }

    def run_sampling(self):
        """Sets up inference run.

        All outputs are written to
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """
        all_sample_lengths = range(
            self.sample_conf.min_length,
            self.sample_conf.max_length+1,
            self.sample_conf.length_step
        )
        for sample_length in all_sample_lengths:
            length_dir = os.path.join(
                self.output_dir, f'length_{sample_length}')
            os.makedirs(length_dir, exist_ok=True)
            self.log.info(f'Sampling length {sample_length}: {length_dir}')
            for sample_i in range(self.sample_conf.samples_per_length):
                sample_dir = os.path.join(length_dir, f'sample_{sample_i}')
                if os.path.isdir(sample_dir):
                    continue
                os.makedirs(sample_dir, exist_ok=True)
                sample_output = self.sample(sample_length)
                traj_paths = self.save_traj(
                    sample_output['prot_traj'],
                    sample_output['rigid_0_traj'],
                    np.ones(sample_length),
                    output_dir=sample_dir
                )










