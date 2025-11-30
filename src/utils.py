"""
Utility functions for molecular graph processing
"""
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType
from torch_geometric.data import Data


ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H']
HYB_TYPES = [
    HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
    HybridizationType.SP3D, HybridizationType.SP3D2, HybridizationType.UNSPECIFIED
]
ATOM_COMBINED_TYPES = {
    (a, h): i for i, (a, h) in enumerate((x, y) for x in ATOM_TYPES for y in HYB_TYPES)
}

BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
BOND_COMBINED_TYPES = {bt: i for i, bt in enumerate(BOND_TYPES)}


def get_node_feature(atom):
    """Extract enhanced node features from an atom."""
    symbol = atom.GetSymbol()
    hyb = atom.GetHybridization()
    idx = ATOM_COMBINED_TYPES.get((symbol, hyb), len(ATOM_COMBINED_TYPES))
    
    # Chirality features
    chiral_tag = atom.GetChiralTag()
    is_chiral_R = int(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
    is_chiral_S = int(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
    
    # Ring size (0 if not in ring)
    mol = atom.GetOwningMol()
    ring_info = mol.GetRingInfo()
    ring_size = 0
    atom_idx = atom.GetIdx()
    for ring in ring_info.AtomRings():
        if atom_idx in ring:
            ring_size = min(ring_size or 999, len(ring))  # Smallest ring size
            break
    
    features = [
        idx,                           # Combined atom type + hybridization
        atom.GetAtomicNum(),          # Atomic number
        atom.GetDegree(),             # Degree
        atom.GetFormalCharge(),       # Formal charge
        atom.GetTotalNumHs(),         # Total H count (implicit + explicit)
        int(atom.GetIsAromatic()),    # Is aromatic
        int(atom.IsInRing()),         # Is in ring
        is_chiral_R,                  # R chirality
        is_chiral_S,                  # S chirality  
        ring_size,                    # Ring size (0 if not in ring)
        atom.GetTotalValence(),       # Total valence (not deprecated)
    ]
    return torch.tensor(features, dtype=torch.float)


def get_edge_feature(bond):
    """Extract edge features from a bond."""
    bond_type = bond.GetBondType()
    is_conjugated = bond.GetIsConjugated()
    in_ring = bond.IsInRing()
    stereo = int(bond.GetStereo())
    is_aromatic = bond.GetIsAromatic()
    bond_type_idx = BOND_COMBINED_TYPES.get(bond_type, len(BOND_COMBINED_TYPES))
    features = [
        bond_type_idx,
        int(is_conjugated),
        int(in_ring),
        stereo,
        int(is_aromatic),
    ]
    return torch.tensor(features, dtype=torch.float)


def smiles_to_data(smiles, y):
    """Convert SMILES string to PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Don't add explicit Hs - use implicit representation for faster training
    # mol = Chem.AddHs(mol)  # REMOVED for 2-3x speedup
    x = torch.stack([get_node_feature(atom) for atom in mol.GetAtoms()])

    edge_index = []
    edge_attr = []

    if mol.GetNumBonds() == 0:
        return None

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        edge_feature = get_edge_feature(bond)
        edge_attr += [edge_feature, edge_feature]  # Directional edges get the same feature

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr, dim=0)
    y = torch.tensor([y], dtype=torch.float)

    N = sum(atom.GetAtomicNum() for atom in mol.GetAtoms())
    N = torch.tensor([N], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, N=N)

