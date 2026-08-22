"""Training loop, evaluation helpers and threshold tuning for the spectrogram CNN.

Trains :class:`~aerosonar.models.spectrogramCNN.SpectrogramCNN` on the recording-level
split from :mod:`aerosonar.data.dataset`, selects the best checkpoint, tunes the
detection threshold, and writes the weights, the threshold, a per-epoch history and the
training figures.

Validation drives per-epoch evaluation, checkpoint selection and threshold tuning. The
test split is not read here at all, so it remains an out-of-sample estimate for the
evaluation suite.

The confusion, probability-collection and threshold-search helpers are importable and are
reused by :mod:`aerosonar.evaluation`.

Run from the repository root::

    python -m aerosonar.training.trainCNN

Set ``AEROSONAR_SHOW_PLOTS`` to display figures interactively; otherwise they are written
to disk only.
"""
import csv
import torch.optim as optim
import torch.nn as nn
import torch
import time
from pathlib import Path
from aerosonar.models.spectrogramCNN import SpectrogramCNN
from aerosonar.data.dataset import build_dataloaders
from aerosonar.config import load_default_config
from aerosonar.utils.seeding import seed_everything
# plotting must be imported before pyplot — it selects the headless backend.
from aerosonar.utils.plotting import COLORS, finish, new_figure, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AverageMeter(object):
    """Accumulates a running average of a metric.

    Attributes:
        val: The most recently recorded value.
        avg: Mean of all recorded values, weighted by sample count.
        sum: Weighted total.
        count: Number of samples recorded.
    """
    def __init__(self, name, fmt=':f'):
        """Create a meter.

        Args:
            name: Label used when formatting.
            fmt: Format specification for the value and average.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Clear all accumulated statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Record a new value.

        Args:
            val: The value, typically a per-batch mean.
            n: Number of samples it represents, used to weight the average.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Format the current value and running average."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Formats a set of meters as an aligned progress line."""

    def __init__(self, num_batches, meters, prefix=""):
        """Create a progress meter.

        Args:
            num_batches: Total batches, used to size the counter field.
            meters: The :class:`AverageMeter` instances to display.
            prefix: Text prepended to each line.
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """Print the current progress line.

        Args:
            batch: Index of the batch just processed.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """Build a zero-padded ``[current/total]`` counter format."""
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy.

    Args:
        output: Model logits of shape ``(batch, classes)``.
        target: Ground-truth class indices of shape ``(batch,)``.
        topk: The values of k to compute.

    Returns:
        list[torch.Tensor]: Accuracy as a percentage for each k, in the order given.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch, device=None):
    """Run one training epoch.

    Args:
        train_loader: Loader over the training split.
        model: The model, switched to training mode.
        criterion: Loss function.
        optimizer: Optimiser, stepped once per batch.
        epoch: Epoch index, used for the progress prefix.
        device: Compute device. Defaults to :data:`DEVICE`.

    Returns:
        tuple: ``(top1_accuracy, batch_losses, batch_errors)`` where the accuracy is a
        percentage averaged over the epoch and the two lists hold per-batch values.
    """
    device = device or DEVICE
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    losses_list = []
    errors = []
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        errors.append(100 - acc1[0].item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses_list.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.display(i)

    return top1.avg, losses_list, errors

def validate(val_loader, model, criterion, device=None):
    """Evaluate the model over a loader without updating it.

    Args:
        val_loader: Loader to evaluate.
        model: The model, switched to evaluation mode.
        criterion: Loss function.
        device: Compute device. Defaults to :data:`DEVICE`.

    Returns:
        tuple: ``(top1_accuracy, batch_losses, batch_errors)``.
    """
    device = device or DEVICE
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    losses_list = []
    errors = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).long()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            errors.append(100 - acc1[0].item())
            losses_list.append(loss.item())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg, losses_list, errors


def evaluate_with_confusion(model, loader, device, threshold=0.5):
    """Compute the confusion matrix over a loader at one threshold.

    Args:
        model: The model, switched to evaluation mode.
        loader: Loader to evaluate.
        device: Compute device.
        threshold: Drone probability above which a chunk is called a detection.

    Returns:
        tuple: ``(TP, FP, TN, FN)``.
    """
    model.eval()

    TP = FP = TN = FN = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] > threshold).long()

            TP += ((preds == 1) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()

    return TP, FP, TN, FN


def confusion_metrics(TP, FP, TN, FN):
    """Derive accuracy, precision, recall and F1 from a confusion matrix.

    Each metric is zero where its denominator would be, so an empty or degenerate
    matrix does not raise.

    Args:
        TP: True positive count.
        FP: False positive count.
        TN: True negative count.
        FN: False negative count.

    Returns:
        tuple: ``(accuracy, precision, recall, f1)``.
    """
    total = TP + FP + TN + FN
    acc  = (TP + TN) / total if total else 0
    prec = TP / (TP + FP) if (TP + FP) else 0
    rec  = TP / (TP + FN) if (TP + FN) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return acc, prec, rec, f1


def collect_probs(model, loader, device):
    """Score an entire loader.

    Args:
        model: The model, switched to evaluation mode.
        loader: Loader to score.
        device: Compute device.

    Returns:
        tuple: ``(probs, labels)``, both one-dimensional CPU tensors, where ``probs``
        holds the drone-class probability per chunk.
    """
    model.eval()
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, 1].cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


def find_best_threshold(probs, labels, step=0.01):
    """Find the threshold maximising F1 over a probability set.

    Args:
        probs: Drone probability per chunk.
        labels: Ground-truth labels, 1 for drone.
        step: Grid spacing for the sweep, which runs from 0.05 to 0.95.

    Returns:
        tuple: ``(best_threshold, metrics, f1_curve)`` where ``metrics`` holds the
        confusion counts and derived metrics at the chosen threshold, and ``f1_curve``
        is a list of ``(threshold, f1)`` pairs covering the whole sweep.
    """
    thresholds = torch.arange(0.05, 0.95 + step, step)
    f1_per_thresh = []
    best_thresh   = 0.5
    best_f1       = -1.0
    best_metrics  = {}

    for thresh in thresholds:
        t     = thresh.item()
        preds = (probs > t).long()
        TP = ((preds == 1) & (labels == 1)).sum().item()
        TN = ((preds == 0) & (labels == 0)).sum().item()
        FP = ((preds == 1) & (labels == 0)).sum().item()
        FN = ((preds == 0) & (labels == 1)).sum().item()
        _, _, _, cm_f1 = confusion_metrics(TP, FP, TN, FN)
        f1_per_thresh.append((t, cm_f1))
        if cm_f1 > best_f1:
            best_f1      = cm_f1
            best_thresh  = t
            cm_acc, cm_prec, cm_rec, _ = confusion_metrics(TP, FP, TN, FN)
            best_metrics = {
                "threshold": best_thresh,
                "f1": cm_f1, "acc": cm_acc,
                "prec": cm_prec, "rec": cm_rec,
                "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            }

    return best_thresh, best_metrics, f1_per_thresh


#: Column order of the per-epoch history CSV.
HISTORY_COLUMNS = [
    "epoch", "lr", "train_loss", "val_loss", "train_acc", "val_acc",
    "val_cm_acc", "val_prec", "val_rec", "val_f1",
]


def write_history(history, path):
    """Persist the per-epoch training record.

    This history is what makes the training-against-validation loss curve, the primary
    overfitting diagnostic, reproducible after the run has finished.

    Args:
        history: One record per epoch, with the keys named in :data:`HISTORY_COLUMNS`.
        path: Destination CSV. Parent directories are created as needed.

    Returns:
        str: The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history written to {path} ({len(history)} epochs)")
    return str(path)


if __name__ == "__main__":
    config = load_default_config()
    train_config = config.get("training", {})
    LR = train_config.get("lr", 0.00001)
    EPOCHS = train_config.get("epochs", 20)
    BATCH_SIZE = train_config.get("batch_size", 64)
    CLASS_WEIGHTS = train_config.get("class_weights", [0.8, 1.2])
    HISTORY_PATH = Path(config["evaluation"]["reports_dir"]) / "train_history.csv"
    GRAPHS_DIR = config["evaluation"]["graphs_dir"]

    seed_everything(config["data"].get("seed", 42))

    # Validation drives checkpoint selection and threshold tuning. Test is never read
    # here, so it remains an out-of-sample estimate for the evaluation suite.
    train_loader, val_loader, test_loader = build_dataloaders(batch_size=BATCH_SIZE)

    device = DEVICE
    net = SpectrogramCNN().to(device)
    loss_weights = torch.tensor(CLASS_WEIGHTS).to(device)  # upweight drone (minority class)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.AdamW(net.parameters(), LR, [0.9, 0.99], 1e-10)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
    )

    train_losses = []
    train_acc1 = []
    val_losses = []
    val_acc1 = []
    acc = []
    prec = []
    rec = []
    f1 = []
    history = []
    best_f1 = -1.0
    best_state = None
    TP, FP, TN, FN = 0, 0, 0, 0

    for epoch in range(0, EPOCHS):
        train_acc, batch_losses, _ = train(train_loader, net, criterion, optimizer, epoch, device)
        train_acc1.append(train_acc.item())
        train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(train_loss)

        val_acc, batch_losses, _ = validate(val_loader, net, criterion, device)
        val_acc1.append(val_acc.item())
        val_loss = sum(batch_losses) / len(batch_losses)
        val_losses.append(val_loss)

        TP, FP, TN, FN = evaluate_with_confusion(net, val_loader, device, threshold=0.5)
        cm_acc, cm_prec, cm_rec, cm_f1 = confusion_metrics(TP, FP, TN, FN)
        acc.append(cm_acc)
        prec.append(cm_prec)
        rec.append(cm_rec)
        f1.append(cm_f1)
        print(f"[Epoch {epoch}] VAL CM: TP={TP} FP={FP} FN={FN} TN={TN} | "
              f"acc={cm_acc:.3f} prec={cm_prec:.3f} rec={cm_rec:.3f} f1={cm_f1:.3f} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        history.append({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc.item(), "val_acc": val_acc.item(),
            "val_cm_acc": cm_acc, "val_prec": cm_prec, "val_rec": cm_rec, "val_f1": cm_f1,
        })

        if cm_f1 > best_f1:
            best_f1 = cm_f1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        # Step on validation loss: a plateau there, rather than in training loss, is the
        # signal that warrants a rate reduction.
        scheduler.step(val_loss)

    write_history(history, HISTORY_PATH)

    weights_dir = Path(config["paths"]["weights"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "CNN_best.pth"
    torch.save(best_state, weights_path)
    print(f"Model saved to {weights_path}")

    # Tune the threshold on validation. Tuning on test would select the cutoff that
    # flatters the split used to report generalization, so those metrics would no longer
    # be out-of-sample.
    net.load_state_dict(best_state)
    all_probs, all_labels = collect_probs(net, val_loader, device)
    best_thresh, thresh_metrics, f1_curve = find_best_threshold(all_probs, all_labels)

    print(f"\n--- Threshold sweep results (validation split) ---")
    print(f"Best threshold : {best_thresh:.2f}")
    print(f"F1             : {thresh_metrics['f1']:.4f}")
    print(f"Precision      : {thresh_metrics['prec']:.4f}")
    print(f"Recall         : {thresh_metrics['rec']:.4f}")
    print(f"Accuracy       : {thresh_metrics['acc']:.4f}")
    print(f"TP={thresh_metrics['TP']} FP={thresh_metrics['FP']} "
          f"FN={thresh_metrics['FN']} TN={thresh_metrics['TN']}")

    import yaml
    with open(weights_dir / "threshold.yaml", "w") as fh:
        yaml.dump({"detection_threshold": best_thresh}, fh)
    print(f"Threshold saved to {weights_dir / 'threshold.yaml'}")

    graphs_dir = Path(GRAPHS_DIR)
    epoch_range = range(1, EPOCHS + 1)

    new_figure()
    plt.plot(epoch_range, acc,  label='Accuracy',  color=COLORS["accuracy"],  linewidth=2)
    plt.plot(epoch_range, f1,   label='F1-Score',  color=COLORS["f1"],        linewidth=2)
    plt.plot(epoch_range, prec, label='Precision', color=COLORS["precision"], linewidth=2)
    plt.plot(epoch_range, rec,  label='Recall',    color=COLORS["recall"],    linewidth=2)
    plt.title('Validation Performance Metrics (threshold=0.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(range(0, EPOCHS + 1, 2))
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    finish(graphs_dir / 'metrics_graph.png')

    # Training against validation loss: the generalization diagnostic.
    new_figure()
    plt.plot(epoch_range, train_losses, label='Train loss',      color=COLORS["train"], linewidth=2)
    plt.plot(epoch_range, val_losses,   label='Validation loss', color=COLORS["val"],   linewidth=2)
    best_epoch = int(np.argmin(val_losses)) + 1
    plt.axvline(best_epoch, color=COLORS["reference"], linestyle="--",
                label=f"Min val loss @ epoch {best_epoch}")
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.xticks(range(0, EPOCHS + 1, 2))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    finish(graphs_dir / 'loss_curves.png')

    thresh_vals = [t for t, _ in f1_curve]
    f1_vals     = [f for _, f in f1_curve]
    new_figure()
    plt.plot(thresh_vals, f1_vals, color=COLORS["f1"], linewidth=2)
    plt.axvline(best_thresh, color=COLORS["precision"], linestyle="--",
                label=f"Best thresh={best_thresh:.2f} (F1={thresh_metrics['f1']:.3f})")
    plt.title('F1 vs Detection Threshold (validation split)')
    plt.xlabel('Threshold')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    finish(graphs_dir / 'threshold_f1.png')

    plot_confusion_matrix(
        thresh_metrics['TP'], thresh_metrics['FP'],
        thresh_metrics['TN'], thresh_metrics['FN'],
        f'Validation Confusion Matrix (threshold={best_thresh:.2f})',
        graphs_dir / 'confusion_matrix.png',
    )
