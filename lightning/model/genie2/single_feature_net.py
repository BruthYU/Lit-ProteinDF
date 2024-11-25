import torch
from torch import nn

from lightning.data.genie2.encoding import sinusoidal_encoding


class SingleFeatureNet(nn.Module):
	"""
	Single Feature Network.

	This module generates per-residue (single) representations by first 
	concatenating multiple encodings, followed by a linear layer to project
	it to given dimension.
	"""

	def __init__(
		self,
		c_s,
		n_timestep,
		c_pos_emb,
		c_chain_emb,
		c_timestep_emb,
		max_n_res,
		max_n_chain,
	):
		"""
		Args:
			c_s:
				Dimension of per-residue (single) representation.
			n_timestep:
				Total number of diffusion timesteps.
			c_pos_emb:
				Dimension of positional (residue index) embedding.
			c_chain_emb:
				Dimension of chain index embedding.
			c_timestep_emb:
				Dimension of diffusion timestep embedding.
			max_n_res:
				Maximum number of residues.
			max_n_chain:
				Maximum number of chains.
		"""
		super(SingleFeatureNet, self).__init__()
		self.c_s = c_s
		self.n_timestep = n_timestep
		self.c_pos_emb = c_pos_emb
		self.c_chain_emb = c_chain_emb
		self.c_timestep_emb = c_timestep_emb
		self.max_n_res = max_n_res
		self.max_n_chain = max_n_chain

		# Layer for final projection
		self.linear = nn.Linear(
			self.c_pos_emb + self.c_chain_emb + self.c_timestep_emb + 25 + 3,
			self.c_s, bias=False
		)

	def forward(self, ts, timesteps, features):
		"""
		Args:
			ts:
				[B, N] Frames at a given timestep.
			timesteps:
				[B] Diffusion timestep.
			features:
				A batched feature dictionary with a batch size B, where each 
				structure is padded to the maximum sequence length N. It contains 
				the following information
					-	aatype: 
							[B, N, 20] one-hot encoding on amino acid types
					-	num_chains: 
							[B, 1] number of chains in the structure
					-	num_residues: 
							[B, 1] number of residues in the structure
					-	num_residues_per_chain: 
							[B, 1] an array of number of residues by chain
					-	atom_positions: 
							[B, N, 3] an array of Ca atom positions
					-	residue_mask: 
							[B, N] residue mask to indicate which residue position is masked
					-	residue_index: 
							[B, N] residue index (started from 0)
					-	chain_index: 
							[B, N] chain index (started from 0)
					-	fixed_sequence_mask: 
							[B, N] mask to indicate which residue contains conditional
							sequence information
					-	fixed_structure_mask: 
							[B, N, N] mask to indicate which pair of residues contains
							conditional structural information
					-	fixed_group:
							[B, N] group index to indicate which group the residue belongs to
							(useful for specifying multiple functional motifs)
					-	interface_mask:
							[B, N] deprecated and set to all zeros.

		Returns:
			[B, N, c_s] Single representation.
		"""

		# Postional (residue index) encoding	
		# Shape: [B, N, c_pos_emb]
		pos_emb = sinusoidal_encoding(
			features['residue_index'],
			self.max_n_res,
			self.c_pos_emb
		)

		# Chain index embedding
		# Shape: [B, N, c_chain_emb]
		chain_emb = sinusoidal_encoding(
			features['chain_index'],
			self.max_n_chain,
			self.c_chain_emb
		)

		# Diffusion timestep embedding
		# Shape: [B, N, c_timestep_emb]
		s = timesteps.unsqueeze(-1).repeat(1, ts.shape[1])
		timestep_emb = sinusoidal_encoding(s, self.n_timestep, self.c_timestep_emb)

		# Masks
		# Shape: [B, N]
		fixed_sequence_mask = features['fixed_sequence_mask']
		interface_mask = features['interface_mask']

		# Residue type embedding
		# Shape: [B, N, 20]
		aatype_emb = features['aatype']
		aatype_emb = aatype_emb * fixed_sequence_mask.unsqueeze(-1)

		# Project to given single representation dimension
		# Shape: [B, N, c_s]
		return self.linear(torch.cat([
			pos_emb,
			chain_emb,
			timestep_emb,
			aatype_emb,
			fixed_sequence_mask.unsqueeze(-1),
			fixed_sequence_mask.unsqueeze(-1),
			interface_mask.unsqueeze(-1)
		], dim=-1)) * features['residue_mask'].unsqueeze(-1)