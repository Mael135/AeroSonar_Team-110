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
from pathlib import Path
import numpy as np
from torchaudio import transforms

# --- Constants ---
METADATA_PATH = 'data/processed/metadata.csv'
TENSOR_DIR = 'data/processed'
BATCH_SIZE = 64
TRAIN_PART = 0.8

# --- 1. Cross-session background mixing ---
class CrossSessionMixer:
    """Synthesizes 'this signal, heard against a different recording session's background' by
    additively mixing in a no-drone power-spectrogram from a different file/location. Always
    mixes in a no-drone (target=0) sample regardless of the anchor's label, so the label never
    changes — it only breaks the location<->label confound (e.g. drone-over-room-background
    chunks that were never actually recorded). Pool is restricted to the train split only.
    """
    def __init__(self, train_meta: pd.DataFrame, data_dir, mix_prob=0.5, mix_gain_db_range=(-6.0, 6.0)):
        self.background_rows = train_meta[train_meta['target'] == 0].reset_index(drop=True)
        self.data_dir = data_dir
        self.mix_prob = mix_prob
        self.mix_gain_db_range = mix_gain_db_range

    def _sample_background(self, exclude_file_id, exclude_location=None):
        pool = self.background_rows[self.background_rows['file_id'] != exclude_file_id]
        if exclude_location is not None:
            other_location = pool[pool['location'] != exclude_location]
            if len(other_location):
                pool = other_location
        if pool.empty:
            return None
        row = pool.sample(n=1).iloc[0]
        return torch.load(os.path.join(self.data_dir, row['filename']), weights_only=True)

    def mix(self, spec: torch.Tensor, file_id, location=None) -> torch.Tensor:
        if random.random() > self.mix_prob:
            return spec
        bg = self._sample_background(exclude_file_id=file_id, exclude_location=location)
        if bg is None:
            return spec
        gain_db = random.uniform(*self.mix_gain_db_range)
        anchor_power = 10 ** (spec / 10.0)
        bg_power = 10 ** ((bg + gain_db) / 10.0)
        mixed_power = (anchor_power + bg_power).clamp_min(1e-10)
        return 10 * torch.log10(mixed_power)


# --- 2. Augmentation Class ---
class TrainAugment:
    def __init__(self, mixer: CrossSessionMixer = None, freq_mask=10, time_mask=8,
                 gain_range=(-6.0, 6.0),
                 time_shift_frac=0.10):
        self.mixer = mixer
        self.freq_mask = transforms.FrequencyMasking(freq_mask_param=freq_mask)
        self.time_mask = transforms.TimeMasking(time_mask_param=time_mask)
        self.gain_range = gain_range
        self.time_shift_frac = time_shift_frac

    def __call__(self, x: torch.Tensor, file_id=None, location=None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self.mixer is not None and file_id is not None:
            x = self.mixer.mix(x, file_id, location)

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
    """Wraps a Subset to apply augmentation only to the training set. `meta` must be the
    per-row metadata (target/file_id/location) for `subset`, in the same order, so the
    transform can look up the anchor's session info for cross-session mixing."""
    def __init__(self, subset, meta=None, transform=None):
        self.subset = subset
        self.meta = meta.reset_index(drop=True) if meta is not None else None
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            if self.meta is not None:
                row = self.meta.iloc[index]
                x = self.transform(x, file_id=row['file_id'], location=row.get('location'))
            else:
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

    expanded_path = Path(metadata_path).parent / "expanded_metadata.csv"
    location_col  = pd.read_csv(expanded_path)[["filename", "location"]]
    full_meta     = full_base_dataset.metadata.merge(location_col, on="filename", how="left")
    assert len(full_meta) == len(full_base_dataset.metadata)

    # Recording-level split — prevents data leakage between train and test.
    # Splitting on chunks would allow chunks from the same WAV file to appear
    # in both splits, inflating test metrics.
    unique_file_ids = full_meta['file_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_file_ids)

    train_count    = int(train_part * len(unique_file_ids))
    train_file_ids = set(unique_file_ids[:train_count])
    test_file_ids  = set(unique_file_ids[train_count:])

    file_id_col   = full_meta['file_id']
    train_indices = file_id_col[file_id_col.isin(train_file_ids)].index.tolist()
    test_indices  = file_id_col[file_id_col.isin(test_file_ids)].index.tolist()

    train_subset = Subset(full_base_dataset, train_indices)
    test_subset  = Subset(full_base_dataset, test_indices)

    # train_meta rows are aligned 1:1 with train_subset (both ordered by train_indices),
    # so the mixer only ever draws backgrounds from train recordings — no test leakage.
    train_meta = full_meta.iloc[train_indices].reset_index(drop=True)
    mixer = CrossSessionMixer(train_meta, tensor_dir, mix_prob=0.5, mix_gain_db_range=(-6.0, 6.0))
    print(f"Cross-session mix pool: {len(mixer.background_rows)} no-drone chunks "
          f"across {mixer.background_rows['location'].nunique()} locations")

    train_transforms = TrainAugment(
        mixer=mixer,
        freq_mask=16,
        time_mask=12,
        gain_range=(-6.0, 6.0),
        time_shift_frac=0.10,
    )

    train_dataset = ApplyTransform(train_subset, meta=train_meta, transform=train_transforms)
    test_dataset  = test_subset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    test_meta = full_meta.iloc[test_indices]
    print(f"Recording-level split: {len(train_file_ids)} recordings → train | "
          f"{len(test_file_ids)} recordings → test")
    print(f"Train chunks: {len(train_dataset)} "
          f"({(train_meta['target']==1).sum()} drone, {(train_meta['target']==0).sum()} ambience)")
    print(f"Test  chunks: {len(test_dataset)} "
          f"({(test_meta['target']==1).sum()} drone, {(test_meta['target']==0).sum()} ambience)")

    return train_loader, test_loader


if __name__ == "__main__":
    build_dataloaders()