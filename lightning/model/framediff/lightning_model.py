import inspect
import random

import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import os
import torch.nn as nn
import ipdb
from lightning.model.framediff import score_network
from lightning.data.framediff import se3_diffuser
from evaluate.openfold.utils import rigid_utils as ru
from preprocess.tools import all_atom

class framediff_Lightning_Model(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self.model_conf = conf.model
        self.frame_conf = conf.frame
        self.exp_conf = conf.experiment
        self.diffuser = se3_diffuser.SE3Diffuser(self.frame_conf)
        self.model = score_network.ScoreNetwork(self.model_conf, self.diffuser)

    def forward(self, batch, cond):
        model_out = self.model(batch)
        return model_out

    def training_step(self, batch, batch_idx, **kwargs):
        loss, aux_data = self.loss_fn(batch)
        self.log("global_step", self.global_step, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_data = self.loss_fn(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def get_schedular(self, optimizer, lr_scheduler='onecycle'):
        if lr_scheduler == 'step':
            scheduler = lrs.StepLR(optimizer,
                                   step_size=self.exp_conf.lr_decay_steps,
                                   gamma=self.exp_conf.lr_decay_rate)
        elif lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer,
                                              T_max=self.exp_conf.lr_decay_steps,
                                              eta_min=self.exp_conf.lr_decay_min_lr)
        elif lr_scheduler == 'onecycle':
            scheduler = lrs.OneCycleLR(optimizer, max_lr=self.exp_conf.learning_rate, steps_per_epoch=self.exp_conf.steps_per_epoch,
                                       epochs=self.exp_conf.num_epoch, three_phase=False)

        elif lr_scheduler == 'LambdaLR':
            def lr_lambda(current_step):
                warmup_steps = self.exp_conf.warmup_steps
                if current_step < self.exp_conf.warmup_steps:
                    # Linearly increase learning rate
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # After warmup, apply other schedule, e.g., constant
                    return 1.0

            scheduler = lrs.LambdaLR(optimizer, lr_lambda=lr_lambda)
            # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)

        else:
            raise ValueError('Invalid lr_scheduler type!')

        return scheduler

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.exp_conf.learning_rate, betas=(0.9, 0.999),
                                      weight_decay=self.exp_conf.weight_decay)
        schecular = self.get_schedular(optimizer, self.exp_conf.lr_scheduler)
        return [optimizer], [{"scheduler": schecular, "interval": "step"}]

    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        if self.model_conf.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad:
                batch = self.self_conditioning(batch)
        model_out = self.model(batch)
        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']
        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        gt_rot_score = batch['rot_score']
        gt_trans_score = batch['trans_score']
        rot_score_scaling = batch['rot_score_scaling']
        trans_score_scaling = batch['trans_score_scaling']
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

        '''Translation score loss'''
        trans_score_mse = (gt_trans_score - pred_trans_score)**2 * loss_mask[..., None]
        trans_score_loss = torch.sum(
            trans_score_mse / trans_score_scaling[:, None, None]**2,
            dim=(-1,-2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        '''Translation x0 loss'''
        gt_trans_x0 = batch['rigids_0'][..., 4:] * self.exp_conf.coordinate_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] * self.exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0)**2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = (
            trans_score_loss * (batch['t'] > self.exp_conf.trans_x0_threshold)
            + trans_x0_loss * (batch['t'] <= self.exp_conf.trans_x0_threshold)
        )
        trans_loss *= self.exp_conf.trans_loss_weight
        trans_loss *= int(self._diff_conf.diffuse_trans)

        '''Rotation loss'''
        if self.exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis)**2 * loss_mask[..., None]
            axis_loss = torch.sum(
                axis_loss, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle)**2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self.exp_conf.rot_loss_weight
            angle_loss *= batch['t'] > self.exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self.exp_conf.rot_loss_weight
            rot_loss *= batch['t'] > self.exp_conf.rot_loss_t_threshold
        rot_loss *= int(self._diff_conf.diffuse_rot)

        '''Backbone atom loss'''
        pred_atom37 = model_out['atom37'][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(torch.float32))
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(
            gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37)**2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self.exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch['t'] < self.exp_conf.bb_atom_loss_t_filter
        bb_atom_loss *= self.exp_conf.aux_loss_weight

        '''Pairwise distance loss'''
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res*5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res*5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res*5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        '''No loss on anything >6A'''
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        dist_mat_loss *= self.exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch['t'] < self.exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self.exp_conf.aux_loss_weight

        '''Final loss'''
        final_loss = (
            rot_loss
            + trans_loss
            + bb_atom_loss
            + dist_mat_loss
        )

        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)


        aux_data = {
            'batch_train_loss': final_loss,
            'batch_rot_loss': rot_loss,
            'batch_trans_loss': trans_loss,
            'batch_bb_atom_loss': bb_atom_loss,
            'batch_dist_mat_loss': dist_mat_loss,
            'total_loss': normalize_loss(final_loss),
            'rot_loss': normalize_loss(rot_loss),
            'trans_loss': normalize_loss(trans_loss),
            'bb_atom_loss': normalize_loss(bb_atom_loss),
            'dist_mat_loss': normalize_loss(dist_mat_loss),
            'examples_per_step': torch.tensor(batch_size),
            'res_length': torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data


