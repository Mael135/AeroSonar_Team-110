import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from pathlib import Path
import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback


# ============================================================
# Helper 1: Convert spectrogram values to dB (log scale)
# ============================================================
def _to_db(specs: np.ndarray, spec_kind: str = "power", eps: float = 1e-10) -> np.ndarray:
    specs = np.asarray(specs)
    if spec_kind == "db":
        return specs.astype(np.float32)
    specs = np.maximum(specs, eps)
    if spec_kind == "magnitude":
        return (20.0 * np.log10(specs)).astype(np.float32)
    if spec_kind == "power":
        return (10.0 * np.log10(specs)).astype(np.float32)
    raise ValueError("spec_kind must be one of: 'power', 'magnitude', 'db'")


# ============================================================
# Helper 2: Decide which label is the "positive" class (Drone)
# ============================================================
def _infer_positive_encoded_label(classes, positive_class=None):
    classes = np.asarray(classes)

    if positive_class is not None:
        for i, c in enumerate(classes):
            if c == positive_class:
                return i, c
        pc = str(positive_class).strip().lower()
        for i, c in enumerate(classes):
            if str(c).strip().lower() == pc:
                return i, c
        raise ValueError(f"positive_class={positive_class} not found in classes={list(classes)}")

    try:
        vals = set(classes.tolist())
        if vals == {0, 1}:
            pos = int(np.where(classes == 1)[0][0])
            return pos, classes[pos]
    except Exception:
        pass

    lowered = [str(c).lower() for c in classes]
    drone_candidates = []
    for i, s in enumerate(lowered):
        if "drone" in s and ("non" not in s) and ("not" not in s):
            drone_candidates.append(i)
    if len(drone_candidates) == 1:
        pos = drone_candidates[0]
        return pos, classes[pos]

    for i, s in enumerate(lowered):
        if "non-drone" in s or "nondrone" in s or ("non" in s and "drone" in s):
            if len(classes) == 2:
                pos = 1 - i
                return pos, classes[pos]

    if len(classes) == 2:
        return 1, classes[1]

    return None, None


def _count_labels(y):
    vals, cnts = np.unique(y, return_counts=True)
    return dict(zip(vals.tolist(), cnts.tolist()))


# ============================================================
# Group-stratified splitting helpers (prevents "val has only 1 class")
# ============================================================
def _group_label_map(y, groups):
    """Map each group -> single label. Raises if a group contains mixed labels."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    m = {}
    for g in np.unique(groups):
        ys = np.unique(y[groups == g])
        if len(ys) != 1:
            raise ValueError(
                f"Group {g} has mixed labels {ys.tolist()} (group collision). Fix group_ids."
            )
        m[g] = int(ys[0])
    return m


def _stratified_group_split_indices(y, groups, test_size, random_state):
    """Split sample indices by splitting UNIQUE groups with stratification on group label."""
    groups = np.asarray(groups)
    y = np.asarray(y)

    g2y = _group_label_map(y, groups)
    uniq_groups = np.array(list(g2y.keys()))
    group_labels = np.array([g2y[g] for g in uniq_groups])

    # Stratify at the GROUP level
    g_train, g_test = train_test_split(
        uniq_groups,
        test_size=test_size,
        random_state=random_state,
        stratify=group_labels,
    )

    train_idx = np.where(np.isin(groups, g_train))[0]
    test_idx = np.where(np.isin(groups, g_test))[0]
    return train_idx, test_idx


# ============================================================
# Main training function
# ============================================================
def train_drone_svm_v5(
    spectrograms,
    labels,
    *,
    spec_kind: str = "power",
    n_components=0.95,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    group_ids=None,
    optimize_metric: str = "recall",
    positive_class=None,
    save_path: str = "drone_final_pipeline.pkl",
    show_plot: bool = True,
    manual_params=None,
):
    specs_array = np.asarray(spectrograms)
    if specs_array.ndim != 3:
        raise ValueError(f"spectrograms must have shape (N,H,W). Got {specs_array.shape}")

    y_raw = np.asarray(labels)
    if len(specs_array) != len(y_raw):
        raise ValueError(f"Mismatch: {len(specs_array)} spectrograms vs {len(y_raw)} labels")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("Need at least 2 classes.")

    pos_idx, pos_label_original = _infer_positive_encoded_label(classes, positive_class=positive_class)

    # Preprocess spectrograms
    specs_db = _to_db(specs_array, spec_kind=spec_kind)
    X = specs_db.reshape(len(specs_db), -1).astype(np.float32)

    # -----------------------------
    # Split (group-stratified if groups exist)
    # -----------------------------
    using_groups = group_ids is not None
    if using_groups:
        groups = np.asarray(group_ids)
        if len(groups) != len(y):
            raise ValueError("group_ids must have same length as labels")

        # Group-stratified split: TrainVal vs Test
        trainval_idx, test_idx = _stratified_group_split_indices(
            y, groups, test_size=test_size, random_state=random_state
        )
        X_trainval, y_trainval, g_trainval = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Group-stratified split: Train vs Val (within trainval)
        train_idx_rel, val_idx_rel = _stratified_group_split_indices(
            y_trainval, g_trainval, test_size=val_size, random_state=random_state + 1
        )
        X_train, y_train, g_train = X_trainval[train_idx_rel], y_trainval[train_idx_rel], g_trainval[train_idx_rel]
        X_val, y_val = X_trainval[val_idx_rel], y_trainval[val_idx_rel]

        # CV folds by GROUPS (must be <= number of unique groups)
        n_unique_groups = len(np.unique(g_train))
        if n_unique_groups < 2:
            raise ValueError("Need at least 2 unique groups for group-based CV.")
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(n_splits=min(5, n_unique_groups))

    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, random_state=random_state + 1, stratify=y_trainval
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Print split label counts (IMPORTANT)
    print("Train counts:", _count_labels(y_train))
    print("Val counts  :", _count_labels(y_val))
    print("Test counts :", _count_labels(y_test))

    # Safety: ensure all splits have both classes
    if n_classes == 2:
        for name, yy in [("TRAIN", y_train), ("VAL", y_val), ("TEST", y_test)]:
            if len(np.unique(yy)) < 2:
                raise ValueError(
                    f"{name} split has only one class. "
                    f"Check group_ids construction and make sure both drone/non-drone recordings exist."
                )

    # Scoring
    metric = optimize_metric.strip().lower()
    if n_classes == 2:
        if pos_idx is None:
            pos_idx = 1
        if metric == "recall":
            scorer = make_scorer(recall_score, pos_label=pos_idx, zero_division=0)
        elif metric == "f1":
            scorer = make_scorer(f1_score, pos_label=pos_idx)
        else:
            raise ValueError("optimize_metric must be 'recall' or 'f1' for binary problems.")
    else:
        scorer = "recall_macro" if metric == "recall" else "f1_macro"

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, svd_solver="full")),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=False, cache_size=512)),
    ])

    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.001],
    }

    print(f"Dataset: {len(X)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print("Classes (encoder order):", [str(c) for c in classes])
    if n_classes == 2:
        print(f"Positive (Drone) class assumed: {pos_label_original} -> encoded as {pos_idx}")
        print(f"Optimizing metric: {optimize_metric}")
    if using_groups:
        print("[INFO] Using group-stratified split + GroupKFold to prevent recording leakage.")

    # -----------------------------
    # Choose hyperparameters: manual OR GridSearchCV
    # -----------------------------
    if manual_params is not None:
        best_params = {
            "svm__C": float(manual_params.get("svm__C", manual_params.get("C", 1.0))),
            "svm__gamma": manual_params.get("svm__gamma", manual_params.get("gamma", "scale")),
        }
        print("\n[INFO] Skipping GridSearchCV. Using manual params:", best_params)

        best_cv_model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, svd_solver="full")),
            ("svm", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=False,
                cache_size=512,
                C=best_params["svm__C"],
                gamma=best_params["svm__gamma"],
            )),
        ])
        best_cv_model.fit(X_train, y_train)

    else:
        grid = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        if using_groups:
            grid.fit(X_train, y_train, groups=g_train)
        else:
            grid.fit(X_train, y_train)

        best_params = grid.best_params_
        print("\nBest CV params:", best_params)
        print("Best CV score:", grid.best_score_)
        best_cv_model = grid.best_estimator_

    # Validation
    y_val_pred = best_cv_model.predict(X_val)
    print("\n--- VALIDATION SET ---")
    print(f"Val Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(classification_report(
        y_val, y_val_pred,
        labels=np.arange(n_classes),
        target_names=[str(c) for c in classes],
        zero_division=0
    ))
    if n_classes == 2:
        print(f"Val Drone Recall: {recall_score(y_val, y_val_pred, pos_label=pos_idx, zero_division=0):.4f}")
        print(f"Val Drone Precision: {precision_score(y_val, y_val_pred, pos_label=pos_idx, zero_division=0):.4f}")

    # Final refit on Train+Val with probability=True
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, svd_solver='full')),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            cache_size=512,
            C=best_params["svm__C"],
            gamma=best_params["svm__gamma"],
        )),
    ])

    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    final_pipe.fit(X_fit, y_fit)

    # Test
    y_test_pred = final_pipe.predict(X_test)
    print("\n--- FINAL TEST SET ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(
        y_test, y_test_pred,
        labels=np.arange(n_classes),
        target_names=[str(c) for c in classes],
        zero_division=0
    ))
    if n_classes == 2:
        print(f"Test Drone Recall: {recall_score(y_test, y_test_pred, pos_label=pos_idx, zero_division=0):.4f}")
        print(f"Test Drone Precision: {precision_score(y_test, y_test_pred, pos_label=pos_idx, zero_division=0):.4f}")
        print(f"Test Drone F1: {f1_score(y_test, y_test_pred, pos_label=pos_idx):.4f}")

    cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(n_classes))
    if show_plot:
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Test)")
        plt.colorbar()
        ticks = np.arange(n_classes)
        plt.xticks(ticks, [str(c) for c in classes], rotation=45, ha="right")
        plt.yticks(ticks, [str(c) for c in classes])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()

    payload = {
        "model": final_pipe,
        "label_encoder": le,
        "classes_": classes,
        "best_params": best_params,
        "spec_kind": spec_kind,
        "n_components": n_components,
        "optimize_metric": optimize_metric,
        "positive_class_original": pos_label_original,
        "positive_class_encoded": pos_idx,
    }
    joblib.dump(payload, save_path)
    print(f"\nSaved to: {save_path}")
    return payload


# ============================================================
# Dataset loader for your processed .pt + CSV files
#   FIXES:
#   - group_ids includes target to avoid collisions
# ============================================================
def load_processed_dataset(processed_dir="data/processed", use_groups=True):
    processed_dir = Path(processed_dir).expanduser().resolve()
    meta_path = processed_dir / "metadata.csv"
    expanded_path = processed_dir / "expanded_metadata.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run process_data() first.")

    df = pd.read_csv(meta_path)
    if "filename" not in df.columns or "target" not in df.columns:
        raise ValueError("metadata.csv must contain: filename,target")

    group_ids = None
    if use_groups and expanded_path.exists():
        exp = pd.read_csv(expanded_path)
        df = df.merge(exp, on="filename", how="left")

        needed = ["date", "location", "noise_level", "noise_type", "gain", "sr", "duration", "num", "target"]
        if all(c in df.columns for c in needed):
            # IMPORTANT: include target to prevent collisions between drone/ambience recordings
            group_ids = (
                "y" + df["target"].astype(str) + "__" +
                df["date"].astype(str) + "__" +
                df["location"].astype(str) + "__" +
                df["noise_level"].astype(str) + "__" +
                df["noise_type"].astype(str) + "__" +
                "g" + df["gain"].astype(str) + "__" +
                "sr" + df["sr"].astype(str) + "__" +
                "dur" + df["duration"].astype(str) + "__" +
                "n" + df["num"].astype(str)
            ).to_numpy()

    filenames = df["filename"].tolist()
    labels = df["target"].astype(int).to_numpy()

    specs = []
    first_shape = None

    for fn in tqdm(filenames, desc="Loading spectrogram .pt files"):
        p = processed_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"Missing spectrogram file: {p}")

        spec = torch.load(p, map_location="cpu")
        if not isinstance(spec, torch.Tensor):
            raise ValueError(f"Unexpected content in {p}: {type(spec)} (expected torch.Tensor)")

        if spec.ndim == 3:
            spec = spec.mean(dim=0)
        if spec.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape {tuple(spec.shape)} in {p}")

        spec_np = spec.detach().cpu().numpy().astype(np.float32)

        if first_shape is None:
            first_shape = spec_np.shape
        elif spec_np.shape != first_shape:
            raise ValueError(f"Inconsistent spectrogram shapes: first={first_shape}, got={spec_np.shape} in {p}")

        specs.append(spec_np)

    spectrograms = np.stack(specs, axis=0)
    return spectrograms, labels, group_ids


# ============================================================
# Main entrypoint
# ============================================================
def main():
    processed_dir = "/Users/guyregev/PycharmProjects/acoustic-drone-detection-Team-110/data/processed"

    spectrograms, labels, group_ids = load_processed_dataset(
        processed_dir=processed_dir,
        use_groups=True
    )

    unique, counts = np.unique(labels, return_counts=True)
    print("Loaded dataset:")
    print("  spectrograms:", spectrograms.shape)
    print("  labels:", labels.shape, " (0=Non-Drone, 1=Drone)")
    print("  label counts:", dict(zip(unique.tolist(), counts.tolist())))
    if group_ids is None:
        print("  group_ids: None")
    else:
        print("  group_ids:", group_ids.shape, "| unique groups:", len(np.unique(group_ids)))

    # If you want to skip the 80 fits and reuse params:
    USE_MANUAL = True
    manual_params = {"svm__C": 0.1, "svm__gamma": "auto"} if USE_MANUAL else None

    payload = train_drone_svm_v5(
        spectrograms=spectrograms,
        labels=labels,
        spec_kind="db",
        group_ids=group_ids,
        optimize_metric="recall",
        positive_class=1,
        save_path=str(Path(processed_dir) / "drone_svm_payload.pkl"),
        show_plot=True,
        manual_params=manual_params,
    )

    print("\nTraining finished.")
    return payload


if __name__ == "__main__":
    main()
