# import torch
# import random
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader, random_split
# import os
# import numpy as np
# from torchaudio import transforms
# METADATA_PATH = 'data\processed\metadata.csv'
# TENSOR_DIR = 'data\processed'
# BATCH_SIZE = 64
# TRAIN_PART = 0.8


# class TrainAugment:
#     def __init__(self, freq_mask=10, time_mask=8,
#                  gain_range=(0.7, 1.3),
#                  time_shift_frac=0.10):
#         self.freq_mask = transforms.FrequencyMasking(freq_mask_param=freq_mask)
#         self.time_mask = transforms.TimeMasking(time_mask_param=time_mask)
#         self.gain_range = gain_range
#         self.time_shift_frac = time_shift_frac

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         # x shape should be (C,F,T) or (F,T). Make it (C,F,T)
#         if x.dim() == 2:
#             x = x.unsqueeze(0)

#         # 1) Random gain (very important)
#         g = random.uniform(*self.gain_range)
#         x = x * g

#         # 2) Random time shift (roll)
#         T = x.shape[-1]
#         max_shift = int(self.time_shift_frac * T)
#         if max_shift > 0:
#             shift = random.randint(-max_shift, max_shift)
#             x = torch.roll(x, shifts=shift, dims=-1)

#         # 3) SpecAugment masks
#         x = self.freq_mask(x)
#         x = self.time_mask(x)

#         return x


# class SpectrogramTensorDataset(Dataset):
#     def __init__(self, metadata_file, data_dir, transform=None, train=False):
#         super().__init__()
#         self.metadata = pd.read_csv(metadata_file)
#         self.data_dir = data_dir
#         self.transform = transform
#         self.train = train

#     def __len__(self):
#         return len(self.metadata)
    
#     def __getitem__(self, idx):
#         file_name = self.metadata.iloc[idx, 0] 
#         file_path = os.path.join(self.data_dir, file_name)
        
#         sample = torch.load(file_path)
        
#         label = self.metadata.iloc[idx, 1]
#         label = torch.tensor(label, dtype=torch.long)

#         if self.transform:
#             sample = self.transform(sample)

#         return sample, label
    


# # dataset = SpectrogramTensorDataset(metadata_file=METADATA_PATH, data_dir=TENSOR_DIR)

# # unique_ids = dataset.metadata['file_id'].unique()
# # np.random.seed(42) # For reproducibility
# # np.random.shuffle(unique_ids)

# # train_count = int(TRAIN_PART * len(unique_ids))
# # train_ids = unique_ids[:train_count]
# # test_ids = unique_ids[train_count:]

# # train_indices = dataset.metadata.index[dataset.metadata['file_id'].isin(train_ids)].tolist()
# # test_indices = dataset.metadata.index[dataset.metadata['file_id'].isin(test_ids)].tolist()

# # train_dataset = torch.utils.data.Subset(dataset, train_indices)
# # test_dataset = torch.utils.data.Subset(dataset, test_indices)

# # # train_size = int(TRAIN_PART * len(dataset))
# # # test_size = int(len(dataset) - train_size)
# # # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# train_transforms = TrainAugment(freq_mask=16, time_mask=12, gain_range=(0.7,1.3), time_shift_frac=0.10)

# train_base = SpectrogramTensorDataset(metadata_file=METADATA_PATH, data_dir=TENSOR_DIR, train=True, transform=train_transforms)
# test_base = SpectrogramTensorDataset(metadata_file=METADATA_PATH, data_dir=TENSOR_DIR, train=False)

# # 2. Split logic (remains the same)
# unique_ids = train_base.metadata['file_id'].unique()
# np.random.seed(42)
# np.random.shuffle(unique_ids)

# train_count = int(TRAIN_PART * len(unique_ids))
# train_ids = unique_ids[:train_count]
# test_ids = unique_ids[train_count:]

# train_indices = train_base.metadata.index[train_base.metadata['file_id'].isin(train_ids)].tolist()
# test_indices = test_base.metadata.index[test_base.metadata['file_id'].isin(test_ids)].tolist()

# # 3. Create subsets pointing to the CORRECT base dataset
# train_dataset = torch.utils.data.Subset(train_base, train_indices)
# test_dataset = torch.utils.data.Subset(test_base, test_indices)

# # 4. DataLoaders (remain the same)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)





























import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
from torchaudio import transforms

# --- Constants ---
METADATA_PATH = 'data/processed/metadata.csv'
TENSOR_DIR = 'data/processed'
BATCH_SIZE = 64
TRAIN_PART = 0.8

# --- 1. Augmentation Class ---
class TrainAugment:
    def __init__(self, freq_mask=10, time_mask=8,
                 gain_range=(-6.0, 6.0),
                 time_shift_frac=0.10):
        self.freq_mask = transforms.FrequencyMasking(freq_mask_param=freq_mask)
        self.time_mask = transforms.TimeMasking(time_mask_param=time_mask)
        self.gain_range = gain_range
        self.time_shift_frac = time_shift_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Additive dB offset — correct way to simulate volume/distance variation
        # on a log-mel spectrogram (multiplicative gain in linear = additive in dB)
        db_offset = random.uniform(*self.gain_range)
        x = x + db_offset

        # Random time shift (roll)
        T = x.shape[-1]
        max_shift = int(self.time_shift_frac * T)
        if max_shift > 0:
            shift = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=shift, dims=-1)

        # SpecAugment masks
        x = self.freq_mask(x)
        x = self.time_mask(x)

        return x

# --- 2. Base Dataset Class ---
class SpectrogramTensorDataset(Dataset):
    def __init__(self, metadata_file, data_dir, transform=None):
        super().__init__()
        self.metadata = pd.read_csv(metadata_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.data_dir, row['filename'])
        sample = torch.load(file_path, weights_only=True)
        label = torch.tensor(row['target'], dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# --- 3. The Transform Wrapper ---
class ApplyTransform(Dataset):
    """Wraps a Subset to apply augmentation only to the training set."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# --- 4. DataLoader Factory ---

def build_dataloaders(
    metadata_path=METADATA_PATH,
    tensor_dir=TENSOR_DIR,
    batch_size=BATCH_SIZE,
    train_part=TRAIN_PART,
):
    """
    Build train and test DataLoaders using a recording-level split.
    Returns (train_loader, test_loader).
    """
    full_base_dataset = SpectrogramTensorDataset(
        metadata_file=metadata_path,
        data_dir=tensor_dir,
    )

    # Recording-level split — prevents data leakage between train and test.
    # Splitting on chunks would allow chunks from the same WAV file to appear
    # in both splits, inflating test metrics.
    unique_file_ids = full_base_dataset.metadata['file_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_file_ids)

    train_count    = int(train_part * len(unique_file_ids))
    train_file_ids = set(unique_file_ids[:train_count])
    test_file_ids  = set(unique_file_ids[train_count:])

    file_id_col   = full_base_dataset.metadata['file_id']
    train_indices = file_id_col[file_id_col.isin(train_file_ids)].index.tolist()
    test_indices  = file_id_col[file_id_col.isin(test_file_ids)].index.tolist()

    train_subset = Subset(full_base_dataset, train_indices)
    test_subset  = Subset(full_base_dataset, test_indices)

    train_transforms = TrainAugment(
        freq_mask=16,
        time_mask=12,
        gain_range=(-6.0, 6.0),
        time_shift_frac=0.10,
    )

    train_dataset = ApplyTransform(train_subset, transform=train_transforms)
    test_dataset  = test_subset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    train_meta = full_base_dataset.metadata.iloc[train_indices]
    test_meta  = full_base_dataset.metadata.iloc[test_indices]
    print(f"Recording-level split: {len(train_file_ids)} recordings → train | "
          f"{len(test_file_ids)} recordings → test")
    print(f"Train chunks: {len(train_dataset)} "
          f"({(train_meta['target']==1).sum()} drone, {(train_meta['target']==0).sum()} ambience)")
    print(f"Test  chunks: {len(test_dataset)} "
          f"({(test_meta['target']==1).sum()} drone, {(test_meta['target']==0).sum()} ambience)")

    return train_loader, test_loader


if __name__ == "__main__":
    build_dataloaders()