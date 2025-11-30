"""
Molecular descriptor extraction utilities
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem, rdMolDescriptors


def calculate_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """
    Calculate Morgan (circular) fingerprint.
    
    Args:
        mol: RDKit molecule
        radius: Radius for fingerprint (default: 2)
        n_bits: Number of bits (default: 2048)
    
    Returns:
        numpy array of fingerprint bits
    """
    if mol is None:
        return np.zeros(n_bits)
    
    # Use new MorganGenerator API to avoid deprecation warnings
    try:
        from rdkit.Chem import rdFingerprintGenerator
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = mfpgen.GetFingerprint(mol)
        return np.array(fp)
    except (ImportError, AttributeError):
        # Fallback to old API if newer version not available
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)


def calculate_basic_descriptors(mol):
    """
    Calculate basic molecular descriptors.
    
    Returns dict with:
    - Molecular weight
    - LogP (lipophilicity)
    - Number of H-bond donors/acceptors
    - Topological polar surface area
    - Number of rotatable bonds
    - Number of aromatic rings
    - Fraction of sp3 carbons
    - Number of heavy atoms
    """
    if mol is None:
        return {}
    
    descriptors = {
        # Basic properties
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Crippen.MolLogP(mol),
        
        # Hydrogen bonding
        'num_h_donors': Lipinski.NumHDonors(mol),
        'num_h_acceptors': Lipinski.NumHAcceptors(mol),
        
        # Polar surface area
        'tpsa': Descriptors.TPSA(mol),
        
        # Flexibility
        'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
        
        # Aromaticity
        'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
        'fraction_csp3': Lipinski.FractionCSP3(mol),
        
        # Size
        'num_heavy_atoms': Lipinski.HeavyAtomCount(mol),
        'num_atoms': mol.GetNumAtoms(),
        
        # Rings
        'num_rings': Lipinski.RingCount(mol),
        'num_saturated_rings': Lipinski.NumSaturatedRings(mol),
        'num_aliphatic_rings': Lipinski.NumAliphaticRings(mol),
        
        # Charge
        'formal_charge': Chem.GetFormalCharge(mol),
        
        # Complexity
        'num_heteroatoms': Lipinski.NumHeteroatoms(mol),
        'num_stereocenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    }
    
    return descriptors


def calculate_extended_descriptors(mol):
    """
    Calculate extended molecular descriptors.
    
    Returns dict with more advanced descriptors.
    """
    if mol is None:
        return {}
    
    descriptors = {
        # Molar refractivity
        'molar_refractivity': Crippen.MolMR(mol),
        
        # Complexity measures
        'bertz_ct': Descriptors.BertzCT(mol),  # Complexity
        'chi0': Descriptors.Chi0(mol),  # Connectivity index
        'chi1': Descriptors.Chi1(mol),
        
        # Shape descriptors
        'kappa1': Descriptors.Kappa1(mol),
        'kappa2': Descriptors.Kappa2(mol),
        'kappa3': Descriptors.Kappa3(mol),
        
        # Electronic properties
        'num_valence_electrons': Descriptors.NumValenceElectrons(mol),
        
        # Lipinski's Rule of Five components
        'lipinski_hba': Descriptors.NumHAcceptors(mol),
        'lipinski_hbd': Descriptors.NumHDonors(mol),
        
        # Additional useful descriptors
        'num_bridgehead_atoms': rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        'num_spiro_atoms': rdMolDescriptors.CalcNumSpiroAtoms(mol),
        'num_saturated_carbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        'num_saturated_heterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        'num_aromatic_carbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        'num_aromatic_heterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
    }
    
    return descriptors


def calculate_all_descriptors(smiles, use_fingerprints=True, fp_bits=512):
    """
    Calculate all molecular descriptors from SMILES.
    
    Args:
        smiles: SMILES string
        use_fingerprints: Whether to include Morgan fingerprints
        fp_bits: Number of bits for fingerprint (default: 512 for efficiency)
    
    Returns:
        torch.Tensor of all descriptors concatenated
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        # Return zeros if molecule is invalid
        total_size = (fp_bits if use_fingerprints else 0) + 32  # ~32 regular descriptors
        return torch.zeros(total_size)
    
    descriptor_list = []
    
    # Add fingerprints if requested
    if use_fingerprints:
        fp = calculate_morgan_fingerprint(mol, radius=2, n_bits=fp_bits)
        descriptor_list.append(fp)
    
    # Add basic descriptors
    basic_desc = calculate_basic_descriptors(mol)
    descriptor_list.append(np.array(list(basic_desc.values())))
    
    # Add extended descriptors
    extended_desc = calculate_extended_descriptors(mol)
    descriptor_list.append(np.array(list(extended_desc.values())))
    
    # Concatenate all descriptors
    all_descriptors = np.concatenate(descriptor_list)
    
    return torch.tensor(all_descriptors, dtype=torch.float32)


def get_descriptor_names(use_fingerprints=True, fp_bits=512):
    """
    Get names of all descriptors for reference.
    
    Returns:
        List of descriptor names
    """
    names = []
    
    if use_fingerprints:
        names.extend([f'morgan_fp_{i}' for i in range(fp_bits)])
    
    # Basic descriptor names
    basic_names = [
        'molecular_weight', 'logp', 'num_h_donors', 'num_h_acceptors',
        'tpsa', 'num_rotatable_bonds', 'num_aromatic_rings', 'fraction_csp3',
        'num_heavy_atoms', 'num_atoms', 'num_rings', 'num_saturated_rings',
        'num_aliphatic_rings', 'formal_charge', 'num_heteroatoms', 'num_stereocenters'
    ]
    names.extend(basic_names)
    
    # Extended descriptor names
    extended_names = [
        'molar_refractivity', 'bertz_ct', 'chi0', 'chi1',
        'kappa1', 'kappa2', 'kappa3', 'num_valence_electrons',
        'lipinski_hba', 'lipinski_hbd', 'num_bridgehead_atoms',
        'num_spiro_atoms', 'num_saturated_carbocycles', 'num_saturated_heterocycles',
        'num_aromatic_carbocycles', 'num_aromatic_heterocycles'
    ]
    names.extend(extended_names)
    
    return names


def normalize_descriptors(descriptors):
    """
    Normalize descriptors to have zero mean and unit variance.
    
    Args:
        descriptors: torch.Tensor of shape (n_samples, n_features)
    
    Returns:
        Normalized descriptors, mean, std
    """
    mean = descriptors.mean(dim=0, keepdim=True)
    std = descriptors.std(dim=0, keepdim=True) + 1e-8  # Add small epsilon to avoid division by zero
    
    normalized = (descriptors - mean) / std
    
    return normalized, mean, std


# Summary of descriptor categories
DESCRIPTOR_CATEGORIES = {
    'Morgan Fingerprint (512 bits)': 'Circular fingerprint capturing local structure',
    'Molecular Weight': 'Size of molecule',
    'LogP': 'Lipophilicity (fat-solubility)',
    'H-bond Donors/Acceptors': 'Hydrogen bonding capacity',
    'TPSA': 'Polar surface area (drug-likeness)',
    'Rotatable Bonds': 'Flexibility',
    'Aromatic Rings': 'Aromaticity',
    'Fraction Csp3': 'Saturation level',
    'Heavy Atoms': 'Non-hydrogen atoms',
    'Rings': 'Ring structures',
    'Formal Charge': 'Net charge',
    'Heteroatoms': 'Non-carbon atoms',
    'Stereocenters': 'Chirality',
    'Molar Refractivity': 'Polarizability',
    'Complexity (Bertz CT)': 'Structural complexity',
    'Connectivity Indices': 'Molecular branching',
    'Kappa Indices': 'Shape descriptors',
}


if __name__ == "__main__":
    # Test the descriptor calculation
    test_smiles = "CCOc1ccc(S(=O)(=O)N2CCN(C)CC2)cc1-c1nc2c(=O)[nH]c(=O)n(C)c2[nH]1"
    
    print("Testing molecular descriptor extraction...")
    print(f"SMILES: {test_smiles}\n")
    
    # Calculate descriptors
    descriptors = calculate_all_descriptors(test_smiles, use_fingerprints=True, fp_bits=512)
    
    print(f"Total number of descriptors: {len(descriptors)}")
    print(f"  - Morgan fingerprint: 512 bits")
    print(f"  - Basic descriptors: 16")
    print(f"  - Extended descriptors: 16")
    print(f"  - Total: {len(descriptors)}")
    
    # Show some example values
    mol = Chem.MolFromSmiles(test_smiles)
    basic = calculate_basic_descriptors(mol)
    
    print("\nExample basic descriptor values:")
    for name, value in list(basic.items())[:8]:
        print(f"  {name}: {value:.3f}")

