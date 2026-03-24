import torch
import pandas as pd
import os
from torch_geometric.data import Dataset
from tqdm import tqdm
from utils import smiles_to_data
from descriptors import calculate_all_descriptors

class IC50Dataset_WithDescriptors(Dataset):
    """Dataset for IC50 molecular data with molecular descriptors."""
    
    def __init__(self, csv_path, transform=None, pre_transform=None, oversample=False, 
                 dataset_name=None, use_fingerprints=True, fp_bits=512):
        super().__init__(None, transform, pre_transform)
        self.dataset_name = dataset_name or os.path.basename(csv_path)
        self.use_fingerprints = use_fingerprints
        self.fp_bits = fp_bits
        
        # Try different delimiters and formats
        for sep in [',', ';']:
            try:
                self.df = pd.read_csv(csv_path, sep=sep)
                
                # Find smiles column (case-insensitive)
                smiles_col = None
                ic50_col = None
                
                for col in self.df.columns:
                    col_lower = col.lower()
                    if 'smiles' in col_lower and smiles_col is None:
                        smiles_col = col
                    if 'ic50' in col_lower and 'log' not in col_lower and ic50_col is None:
                        ic50_col = col
                
                if smiles_col and ic50_col:
                    # Rename to standard names
                    self.df = self.df[[smiles_col, ic50_col]].copy()
                    self.df.columns = ['smiles', 'IC50']
                    break
            except:
                continue
        
        # Check if we successfully loaded the data
        if not hasattr(self, 'df') or 'smiles' not in self.df.columns:
            # If IC50 is missing, we might be in inference mode without ground truth
            # But for this class designed for training/eval with known IC50, we usually expect it.
            # However, for pure inference, we might want to allow missing IC50.
            # For now, I'll stick to the original implementation which requires IC50, 
            # and if needed for inference on new molecules without labels, we might need to adjust.
            # But the user said "evaluate feature", implying we might have labels or want to predict.
            # If I look at the original code, it raises ValueError if IC50 is missing.
            # I will keep it as is for now, but maybe I should allow missing IC50 for inference?
            # The user said "inference for the model in another datasets", usually implies we might not have labels.
            # But "evaluate" implies we do.
            # Let's stick to the original code for now to ensure compatibility, and I can modify it if needed for pure prediction.
            if not hasattr(self, 'df') or 'smiles' not in self.df.columns or 'IC50' not in self.df.columns:
                 raise ValueError(f"Could not find 'smiles' and 'IC50' columns in {self.dataset_name}")
        
        self.df.dropna(subset=['smiles', 'IC50'], inplace=True)
        
        # Remove rows with non-numeric IC50 values
        self.df = self.df[pd.to_numeric(self.df['IC50'], errors='coerce').notna()]
        self.df['IC50'] = pd.to_numeric(self.df['IC50'])
        
        # Remove rows with IC50 <= 0 (can't take log)
        self.df = self.df[self.df['IC50'] > 0]
        
        if len(self.df) == 0:
            raise ValueError(f"No valid samples in {self.dataset_name}")
        
        self.df['log_IC50'] = self.df['IC50'].apply(lambda x: torch.log10(torch.tensor(float(x))))
        
        if oversample:
            self.df = self.df.sample(frac=2, replace=True, random_state=42).reset_index(drop=True)
        
        # Create a unique cache filename based on dataset properties
        cache_filename = f"{self.dataset_name}_descriptors"
        cache_filename += f"_fp{fp_bits}" if use_fingerprints else "_nofp"
        if oversample:
            cache_filename += "_os"
            
        cache_dir = os.path.dirname(csv_path)
        cache_path = os.path.join(cache_dir, f"{cache_filename}.pt")
        
        if os.path.exists(cache_path):
            print(f"  Loading cached descriptors for {self.dataset_name} from {cache_path}...")
            self.descriptors = torch.load(cache_path)
            # Ensure cache is valid shape
            if len(self.descriptors) != len(self.df):
                print(f"  Cache length mismatch ({len(self.descriptors)} vs {len(self.df)}). Recalculating...")
                self.descriptors = None
        else:
            self.descriptors = None
            
        if self.descriptors is None:
            print(f"  Calculating descriptors for {self.dataset_name}...")
            descriptor_list = []
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"    {self.dataset_name}", leave=False):
                desc = calculate_all_descriptors(row['smiles'], 
                                                use_fingerprints=use_fingerprints, 
                                                fp_bits=fp_bits)
                descriptor_list.append(desc)
            
            self.descriptors = torch.stack(descriptor_list)
            print(f"  Saving computed descriptors to {cache_path}...")
            torch.save(self.descriptors, cache_path)
        
        # Normalize descriptors
        self.desc_mean = self.descriptors.mean(dim=0, keepdim=True)
        self.desc_std = self.descriptors.std(dim=0, keepdim=True) + 1e-8
        self.descriptors = (self.descriptors - self.desc_mean) / self.desc_std
        
        print(f"  Loaded {self.dataset_name}: {len(self.df)} samples with {self.descriptors.shape[1]} descriptors")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = smiles_to_data(row['smiles'], row['log_IC50'])

        while data is None:
            idx = (idx + 1) % len(self.df)
            row = self.df.iloc[idx]
            data = smiles_to_data(row['smiles'], row['log_IC50'])
        
        # Add descriptors to data object
        data.descriptors = self.descriptors[idx]
        data.descriptor_size = self.descriptors.shape[1]
        
        # Add dataset identifier
        data.dataset_name = self.dataset_name
        return data
