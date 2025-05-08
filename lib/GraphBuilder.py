"""
@File    : GraphBuilder.py
@Time    : 15/1/25 12:42
@Author  : iRinYe
@Contact : iRinYeh@outlook.com
"""
import Bio
import numpy as np
import torch
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist

"""

"""


def pdb_to_graph(pdb_file, max_length, padding_aa, padding_coord, chain_id=None):
	sequence, positions = _extract_positions_and_sequence(pdb_file, chain_id, max_length, padding_aa, padding_coord)
	if not positions:
		return None, None, None
	positions = np.array(positions)
	node_features = torch.tensor(positions, dtype=torch.float)
	edge_index = _build_edge_index(positions, padding_coord)
	sequence_str = ''.join(sequence)
	return node_features, edge_index, sequence_str


def _extract_positions_and_sequence(pdb_file, chain_id, max_length, padding_aa, padding_coord):
	amino_acids = {
		'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
		'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
		'MET', 'ASN', 'PRO', 'GLN', 'ARG',
		'SER', 'THR', 'VAL', 'TRP', 'TYR'
	}
	parser = PDBParser(QUIET=True)
	structure = parser.get_structure('PDB_structure', pdb_file)
	positions = []
	sequence = []
	for model in structure:
		for chain in model:
			if chain_id is None and len(model) > 1:
				raise ValueError(f"Multiple chains found in {pdb_file}, but no chain_id provided.")
			if chain_id is None or chain.id == chain_id:
				for residue in chain:
					if len(sequence) >= max_length:
						break
					res_name = residue.get_resname()
					if res_name in amino_acids and 'CA' in residue:
						seq1 = Bio.PDB.Polypeptide.protein_letters_3to1[res_name]
						sequence.append(seq1)
						ca_atom = residue['CA']
						positions.append(ca_atom.coord)
	sequence, positions = _pad_sequence_and_positions(sequence, positions, max_length, padding_aa, padding_coord)
	return sequence, positions


def _pad_sequence_and_positions(sequence, positions, max_length, padding_aa, padding_coord):
	num_fill = max_length - len(sequence)
	if num_fill > 0:
		sequence.extend([padding_aa] * num_fill)
		positions.extend([np.array(padding_coord)] * num_fill)
	return sequence, positions


def _build_edge_index(positions, padding_coord):
	positions = positions[np.all(positions != padding_coord, axis=1)]
	if len(positions) == 0:
		return None
	distances = cdist(positions, positions)
	edge_indices = np.where(distances <= 5.0)
	edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
	return edge_index