import gzip
import torch
import numpy as np
import json
from preprocess.tools import residue_constants
from lightning.data.genie2.motif_utils import load_motif_spec, sample_motif_mask
from itertools import compress


def pad_np_features(np_features, max_n_chain, max_n_res):
    num_chains = np_features['num_chains']
    num_residues = np_features['num_residues']
    for key in np_features:
        if key == 'num_residues_per_chain':
            np_features[key] = np.concatenate([
                np_features[key],
                np.zeros(max_n_chain - num_chains).astype(np_features[key].dtype)
            ])
        elif key == 'fixed_structure_mask':
            np_features[key] = np.pad(
                np_features[key],
                [
                    (0, max_n_res - num_residues),
                    (0, max_n_res - num_residues)
                ],
                'constant',
                constant_values=0
            ).astype(np_features[key].dtype)
        elif not key.startswith('num'):
            np_features[key] = np.concatenate([
                np_features[key],
                np.zeros((
                    max_n_res - num_residues,
                    *np_features[key].shape[1:]
                )).astype(np_features[key].dtype)
            ])
    return np_features


def create_empty_np_features(lengths):
    num_chains = np.array(len(lengths))
    num_residues = np.sum(lengths)
    num_residues_per_chain = np.array(lengths)

    # Generate
    aatype = np.zeros((num_residues, len(residue_constants.restypes_with_x)))
    atom_positions = np.zeros((num_residues, 3))
    residue_mask = np.ones(num_residues)
    residue_index = np.concatenate([
        np.arange(length)
        for length in lengths
    ])
    chain_index = np.concatenate([
        [idx] * length
        for idx, length in enumerate(lengths)
    ])
    fixed_sequence_mask = np.zeros(num_residues)
    fixed_structure_mask = np.zeros((num_residues, num_residues))
    fixed_group = np.zeros(num_residues)
    interface_mask = np.zeros(num_residues)

    # Create
    np_features = {
        'aatype': aatype.astype(int),
        'num_chains': num_chains.astype(int),
        'num_residues': num_residues.astype(int),
        'num_residues_per_chain': num_residues_per_chain.astype(int),
        'atom_positions': atom_positions.astype(float),
        'residue_mask': residue_mask.astype(int),
        'residue_index': residue_index.astype(int),
        'chain_index': chain_index.astype(int),
        'fixed_sequence_mask': fixed_sequence_mask.astype(bool),
        'fixed_structure_mask': fixed_structure_mask.astype(bool),
        'fixed_group': fixed_group.astype(int),
        'interface_mask': interface_mask.astype(bool)
    }

    return np_features
