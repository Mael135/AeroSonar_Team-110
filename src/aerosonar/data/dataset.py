"""Dataset, augmentation and data-splitting for spectrogram training.

Loads the preprocessed tensors written by :mod:`aerosonar.data.preprocessData` and
exposes them as PyTorch ``DataLoader`` objects over a label-stratified,
recording-level train/validation/test split.

Two properties of the split are load-bearing and should be preserved by any change:

* Splitting is by recording, never by chunk. Adjacent chunks of one recording are
  near-duplicates, so a chunk-level split would place copies of the same audio on both
  sides and inflate held-out metrics.
* Augmentation, including cross-session mixing, is applied to the training split only.
  Validation and test are served unmodified so their metrics describe the input
  distribution the deployed model actually sees.

Run from the repository root to print a split summary::

    python -m aerosonar.data.dataset
"""
import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import os
from pathlib import Path
import numpy as np
from torchaudio import transforms

METADATA_PATH = 'data/processed/metadata.csv'
TENSOR_DIR = 'data/processed'
BATCH_SIZE = 64
TRAIN_PART = 0.70
VAL_PART = 0.15   # test gets the remainder
SEED = 42
SPLIT_REPORT_PATH = 'reports/eval/split_composition.csv'


class CrossSessionMixer:
    """Mixes a background from one recording session into a sample from another.

    Additively combines the anchor spectrogram with a no-drone spectrogram drawn from a
    different recording and, where available, a different location. The mixed-in sample
    is always no-drone regardless of the anchor's label, so the anchor's label is
    unaffected.

    The purpose is to weaken the correlation between recording location and class label
    present in the corpus, by synthesising combinations such as a drone heard over an
    indoor background that were never actually recorded. The background pool is drawn
    from the training split only, so no held-out audio reaches the model.

    Mixing is performed in the power domain, which is the physically correct way to
    combine two independent sources represented as decibel spectrograms.

    Attributes:
        background_rows: Candidate no-drone rows from the training split.
        mix_prob: Probability that any given call mixes rather than passes through.
        mix_gain_db_range: Range of gain offsets applied to the background.
    """

    def __init__(self, train_meta: pd.DataFrame, data_dir, mix_prob=0.5, mix_gain_db_range=(-6.0, 6.0)):
        """Build the background pool.

        Args:
            train_meta: Training-split metadata with ``target``, ``file_id`` and
                ``location`` columns. Only no-drone rows are retained.
            data_dir: Directory holding the ``.pt`` tensors.
            mix_prob: Probability of mixing on each call to :meth:`mix`.
            mix_gain_db_range: Uniform range, in decibels, for the background gain.
        """
        self.background_rows = train_meta[train_meta['target'] == 0].reset_index(drop=True)
        self.data_dir = data_dir
        self.mix_prob = mix_prob
        self.mix_gain_db_range = mix_gain_db_range

    def _sample_background(self, exclude_file_id, exclude_location=None):
        """Draw one no-drone background from a different recording.

        Prefers a different location as well, falling back to same-location candidates
        when excluding the location would empty the pool.

        Args:
            exclude_file_id: Recording to exclude.
            exclude_location: Location to avoid where possible.

        Returns:
            torch.Tensor | None: The background spectrogram, or None if no candidate
            remains.
        """
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
        """Mix a background into a spectrogram, with probability ``mix_prob``.

        Args:
            spec: Anchor decibel spectrogram.
            file_id: The anchor's source recording, excluded from the background pool.
            location: The anchor's location, avoided where possible.

        Returns:
            torch.Tensor: The mixed spectrogram, or ``spec`` unchanged if this call did
            not mix or no background was available.
        """
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


class TrainAugment:
    """Training-time augmentation for decibel mel spectrograms.

    Applies, in order: cross-session background mixing, a random level offset, a random
    circular time shift, and SpecAugment frequency and time masking.

    Uses the ``random`` module and PyTorch's generator, so
    :func:`aerosonar.utils.seeding.seed_everything` is required for a reproducible run.
    """

    def __init__(self, mixer: CrossSessionMixer = None, freq_mask=10, time_mask=8,
                 gain_range=(-6.0, 6.0),
                 time_shift_frac=0.10):
        """Configure the augmentation chain.

        Args:
            mixer: Cross-session mixer. If None, mixing is skipped.
            freq_mask: Maximum width, in mel bands, of the frequency mask.
            time_mask: Maximum width, in frames, of the time mask.
            gain_range: Uniform range, in decibels, for the level offset.
            time_shift_frac: Maximum time shift as a fraction of clip length.
        """
        self.mixer = mixer
        self.freq_mask = transforms.FrequencyMasking(freq_mask_param=freq_mask)
        self.time_mask = transforms.TimeMasking(time_mask_param=time_mask)
        self.gain_range = gain_range
        self.time_shift_frac = time_shift_frac

    def __call__(self, x: torch.Tensor, file_id=None, location=None) -> torch.Tensor:
        """Augment one spectrogram.

        Args:
            x: Spectrogram of shape ``(freq, time)`` or ``(channels, freq, time)``.
                Two-dimensional input gains a leading channel axis.
            file_id: Source recording, required for cross-session mixing.
            location: Source location, used to prefer a different one when mixing.

        Returns:
            torch.Tensor: The augmented spectrogram, shape
            ``(channels, freq, time)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self.mixer is not None and file_id is not None:
            x = self.mixer.mix(x, file_id, location)

        # A multiplicative gain in the linear domain is an additive offset in decibels,
        # so this simulates variation in volume and source distance.
        db_offset = random.uniform(*self.gain_range)
        x = x + db_offset

        T = x.shape[-1]
        max_shift = int(self.time_shift_frac * T)
        if max_shift > 0:
            shift = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=shift, dims=-1)

        x = self.freq_mask(x)
        x = self.time_mask(x)

        return x


class SpectrogramTensorDataset(Dataset):
    """Serves the preprocessed spectrogram tensors listed in a metadata CSV.

    Each item is loaded from its own ``.pt`` file on access rather than held in memory.

    Attributes:
        metadata: The parsed CSV, with ``filename`` and ``target`` columns. Row order
            defines dataset indices.
    """

    def __init__(self, metadata_file, data_dir, transform=None):
        """Open the dataset.

        Args:
            metadata_file: CSV listing ``filename`` and ``target`` per chunk.
            data_dir: Directory holding the ``.pt`` tensors.
            transform: Optional callable applied to each sample. Normally left None
                here and applied via :class:`ApplyTransform` so that augmentation
                reaches the training split only.
        """
        super().__init__()
        self.metadata = pd.read_csv(metadata_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        """Return the number of chunks."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """Load one chunk.

        Args:
            idx: Row index into the metadata.

        Returns:
            tuple: ``(sample, label)`` where ``sample`` is a float spectrogram tensor
            of shape ``(1, n_mels, frames)`` and ``label`` is a ``torch.long`` scalar,
            0 for ambience and 1 for drone.
        """
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.data_dir, row['filename'])
        sample = torch.load(file_path, weights_only=True)
        label = torch.tensor(row['target'], dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class ApplyTransform(Dataset):
    """Wraps a ``Subset`` so augmentation applies to one split only.

    Where metadata is supplied, the recording and location of each sample are passed
    through to the transform, which cross-session mixing requires in order to exclude
    the sample's own recording from the background pool.
    """

    def __init__(self, subset, meta=None, transform=None):
        """Wrap a subset.

        Args:
            subset: The ``Subset`` to wrap.
            meta: Per-row metadata for ``subset`` with ``file_id`` and ``location``
                columns, in the same order as the subset's indices. Misalignment here
                would attribute samples to the wrong recording.
            transform: Callable applied to each sample.
        """
        self.subset = subset
        self.meta = meta.reset_index(drop=True) if meta is not None else None
        self.transform = transform

    def __getitem__(self, index):
        """Load one sample and apply the transform.

        Args:
            index: Index into the wrapped subset.

        Returns:
            tuple: ``(sample, label)``.
        """
        x, y = self.subset[index]
        if self.transform:
            if self.meta is not None:
                row = self.meta.iloc[index]
                x = self.transform(x, file_id=row['file_id'], location=row.get('location'))
            else:
                x = self.transform(x)
        return x, y

    def __len__(self):
        """Return the number of samples in the wrapped subset."""
        return len(self.subset)


def load_joined_metadata(metadata_path=METADATA_PATH):
    """Load the chunk metadata joined with each chunk's recording location.

    Joins ``metadata.csv`` against the ``location`` column of
    ``expanded_metadata.csv``, which sits in the same directory.

    Args:
        metadata_path: Path to ``metadata.csv``.

    Returns:
        pandas.DataFrame: Columns ``filename``, ``target``, ``file_id`` and
        ``location``.

    Raises:
        AssertionError: If the join changes the row count, which would indicate
            duplicate or missing filenames between the two files.
    """
    meta = pd.read_csv(metadata_path)
    expanded_path = Path(metadata_path).parent / "expanded_metadata.csv"
    location_col = pd.read_csv(expanded_path)[["filename", "location"]]
    joined = meta.merge(location_col, on="filename", how="left")
    assert len(joined) == len(meta), "location join changed the row count"
    return joined


def split_file_ids(meta, train_part=TRAIN_PART, val_part=VAL_PART, seed=SEED):
    """Partition recordings into train, validation and test sets.

    The split is at the recording level, so chunks from one source file never straddle
    a boundary, and is stratified by label: each class's recordings are shuffled and
    allocated independently. Stratification matters at this corpus size, where an
    unstratified shuffle of roughly twenty recordings can leave a split with no drone
    recordings at all, making precision and recall undefined.

    Args:
        meta: Metadata with ``file_id`` and ``target`` columns.
        train_part: Fraction of each class's recordings for training.
        val_part: Fraction for validation. Test receives the remainder.
        seed: Seed for the shuffle, making the split reproducible.

    Returns:
        tuple: Three sets of ``file_id`` values, ``(train, val, test)``.

    Raises:
        ValueError: If either class has fewer than three recordings, which cannot be
            divided three ways.
        AssertionError: If the resulting splits overlap or lose a recording.
    """
    rec_level = meta.drop_duplicates("file_id")[["file_id", "target"]].sort_values("file_id")
    rng = np.random.default_rng(seed)

    train_ids, val_ids, test_ids = [], [], []
    for target, group in rec_level.groupby("target"):
        ids = group["file_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            raise ValueError(
                f"class {target} has only {n} recording(s); a three-way split needs at least 3"
            )
        n_train = int(np.floor(train_part * n))
        n_val = max(1, int(round(val_part * n)))
        # Keep at least one recording each for test and train once rounding is applied.
        n_train = min(n_train, n - n_val - 1)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_ids & val_ids), "train/val recording overlap"
    assert not (train_ids & test_ids), "train/test recording overlap"
    assert not (val_ids & test_ids), "val/test recording overlap"
    assert len(train_ids | val_ids | test_ids) == len(rec_level), "recordings lost in the split"
    return train_ids, val_ids, test_ids


def describe_splits(meta, splits, report_path=SPLIT_REPORT_PATH):
    """Summarise and persist the composition of each split.

    Prints a per-split summary, warns about split-and-location combinations containing
    only one class, and writes the full breakdown to CSV. A location represented by a
    single class allows the label to be predicted from the recording environment alone,
    so this table qualifies any metric computed on these splits and belongs alongside
    the reported results.

    Args:
        meta: Metadata with ``file_id``, ``target`` and ``location`` columns.
        splits: Maps split name to its set of ``file_id`` values.
        report_path: Destination CSV. Parent directories are created as needed.

    Returns:
        pandas.DataFrame: One row per split, location and label, with recording and
        chunk counts.
    """
    rows = []
    for name, file_ids in splits.items():
        part = meta[meta["file_id"].isin(file_ids)]
        for (location, target), group in part.groupby(["location", "target"]):
            rows.append({
                "split": name, "location": location, "target": int(target),
                "label": "drone" if target == 1 else "ambience",
                "recordings": group["file_id"].nunique(), "chunks": len(group),
            })
    table = pd.DataFrame(rows)

    for name, file_ids in splits.items():
        part = meta[meta["file_id"].isin(file_ids)]
        print(f"{name:5s}: {len(file_ids):2d} recordings | {len(part):5d} chunks "
              f"({(part['target'] == 1).sum():4d} drone, {(part['target'] == 0).sum():4d} ambience) "
              f"| locations: {sorted(part['location'].dropna().unique())}")

    single_class = [
        f"{r.split}/{r.location}"
        for r in table.itertuples()
        if len(table[(table.split == r.split) & (table.location == r.location)]) == 1
    ]
    if single_class:
        print(f"  NOTE: single-class split x location cells (label shortcut risk): "
              f"{sorted(set(single_class))}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_path, index=False)
    print(f"  Split composition written to {report_path}")
    return table


def build_loaders_from_file_ids(
    full_meta,
    base_dataset,
    train_file_ids,
    val_file_ids,
    test_file_ids,
    tensor_dir=TENSOR_DIR,
    batch_size=BATCH_SIZE,
):
    """Build data loaders for an explicit recording-level split.

    Use this directly when the split is determined externally, for example by
    cross-validation. :func:`build_dataloaders` wraps it with the standard split.

    Augmentation and cross-session mixing are applied to the training loader only.

    Args:
        full_meta: Metadata for the whole dataset, row-aligned with ``base_dataset``.
        base_dataset: The underlying :class:`SpectrogramTensorDataset`.
        train_file_ids: Recordings for training.
        val_file_ids: Recordings for validation.
        test_file_ids: Recordings for test.
        tensor_dir: Directory holding the ``.pt`` tensors, for the mixer.
        batch_size: Batch size for all three loaders.

    Returns:
        tuple: ``(train_loader, val_loader, test_loader)``. Only the training loader
        shuffles.

    Raises:
        AssertionError: If the mixer's background pool contains a held-out recording.
    """
    file_id_col = full_meta['file_id']
    indices = {
        name: file_id_col[file_id_col.isin(ids)].index.tolist()
        for name, ids in (("train", train_file_ids), ("val", val_file_ids), ("test", test_file_ids))
    }

    # train_meta is ordered by train_indices, matching train_subset row for row, which
    # ApplyTransform relies on to look up each sample's recording.
    train_meta = full_meta.iloc[indices["train"]].reset_index(drop=True)
    mixer = CrossSessionMixer(train_meta, tensor_dir, mix_prob=0.5, mix_gain_db_range=(-6.0, 6.0))
    held_out = set(val_file_ids) | set(test_file_ids)
    assert not (set(mixer.background_rows['file_id']) & held_out), \
        "cross-session mixer would leak a held-out recording into training"
    print(f"Cross-session mix pool: {len(mixer.background_rows)} no-drone chunks "
          f"across {mixer.background_rows['location'].nunique()} locations")

    train_transforms = TrainAugment(
        mixer=mixer,
        freq_mask=16,
        time_mask=12,
        gain_range=(-6.0, 6.0),
        time_shift_frac=0.10,
    )

    train_dataset = ApplyTransform(
        Subset(base_dataset, indices["train"]), meta=train_meta, transform=train_transforms
    )
    val_dataset = Subset(base_dataset, indices["val"])
    test_dataset = Subset(base_dataset, indices["test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def build_dataloaders(
    metadata_path=METADATA_PATH,
    tensor_dir=TENSOR_DIR,
    batch_size=BATCH_SIZE,
    train_part=TRAIN_PART,
    val_part=VAL_PART,
    seed=SEED,
    describe=True,
):
    """Build train, validation and test loaders using the standard split.

    The validation split exists so that the detection threshold and the best checkpoint
    can be selected without consulting the test split, which would otherwise cease to
    be an out-of-sample estimate.

    Args:
        metadata_path: Path to ``metadata.csv``.
        tensor_dir: Directory holding the ``.pt`` tensors.
        batch_size: Batch size for all three loaders.
        train_part: Fraction of each class's recordings for training.
        val_part: Fraction for validation. Test receives the remainder.
        seed: Seed for the recording shuffle.
        describe: Whether to print and persist the split composition.

    Returns:
        tuple: ``(train_loader, val_loader, test_loader)``.
    """
    base_dataset = SpectrogramTensorDataset(metadata_file=metadata_path, data_dir=tensor_dir)
    full_meta = load_joined_metadata(metadata_path)

    train_ids, val_ids, test_ids = split_file_ids(full_meta, train_part, val_part, seed)
    if describe:
        describe_splits(full_meta, {"train": train_ids, "val": val_ids, "test": test_ids})

    return build_loaders_from_file_ids(
        full_meta, base_dataset, train_ids, val_ids, test_ids,
        tensor_dir=tensor_dir, batch_size=batch_size,
    )


if __name__ == "__main__":
    build_dataloaders()
