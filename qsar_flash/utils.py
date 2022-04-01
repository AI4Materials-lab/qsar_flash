from typing import Any, Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover


def generate_3d_coords(mol: Chem.rdchem.Mol, random_seed: int = 0) -> Chem.rdchem.Mol:
    """Generate random conformer for molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RdKit molecular object.
        random_seed (int, optional): random seed. Defaults to 0.

    Rises:
        ValueError Conformer not found.

    Returns:
        Chem.rdchem.Mol: RdKit molecular object with conformer.
    """

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=False, randomSeed=random_seed)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    mol = Chem.RemoveHs(mol)
    return mol


def smiles2graph(smiles_string: str) -> Dict[str, Any]:
    """Converts SMILES string to graph Data object.

    Args:
        smiles_string (str): SMILES string.

    Rises:
        ValueError  Conformer not found.

    Returns:
        Dict[str, Any]: graph object.
    """
    salt_remover = SaltRemover(defnFormat="smarts")
    mol = Chem.MolFromSmiles(smiles_string)
    mol = salt_remover.StripMol(mol)
    mol = generate_3d_coords(mol)
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom.GetAtomicNum())
    z = np.array(atom_features_list, dtype=np.int64)
    pos = mol.GetConformer().GetPositions()

    graph: Dict[str, Any] = dict()
    graph["z"] = z
    graph["pos"] = pos
    graph["num_nodes"] = len(z)

    return graph
