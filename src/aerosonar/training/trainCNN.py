import os
import torch.optim as optim
import torch.nn as nn
import torch
import time
from pathlib import Path
from aerosonar.models.spectrogramCNN import SpectrogramCNN
from aerosonar.data.dataset import build_dataloaders
from aerosonar.config import load_default_config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


def train(train_loader, model, criterion, optimizer, epoch):
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

def validate(val_loader, model, criterion):
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

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg, losses_list, errors


def evaluate_with_confusion(model, loader, device, threshold=0.5):
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
    total = TP + FP + TN + FN
    acc  = (TP + TN) / total if total else 0
    prec = TP / (TP + FP) if (TP + FP) else 0
    rec  = TP / (TP + FN) if (TP + FN) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return acc, prec, rec, f1


def collect_probs(model, loader, device):
    """Return drone-class probabilities and ground-truth labels for the full loader."""
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
    """
    Sweep thresholds from 0.05 to 0.95 and return the one that maximises F1.
    Returns (best_threshold, metrics_dict, per_threshold_f1_list).
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




if __name__ == "__main__":
    LR = 0.00002
    EPOCHS = 20

    config = load_default_config()
    train_loader, test_loader = build_dataloaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SpectrogramCNN().to(device)
    loss_weights = torch.tensor([0.8, 1.2]).to(device)  # upweight drone (minority class)
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
    test_losses = []
    test_acc1 = []
    test_error = []
    train_error = []
    acc = []
    prec = []
    rec = []
    f1 = []
    best_f1 = -1.0
    best_state = None
    TP, FP, TN, FN = 0, 0, 0, 0

    for epoch in range(0, EPOCHS):
        acc1, losses, error = train(train_loader, net, criterion, optimizer, epoch)
        train_acc1.append(acc1.item())
        train_losses.append(sum(losses) / len(losses))
        train_error.extend(error)

        acc1, losses, error = validate(test_loader, net, criterion)
        test_acc1.append(acc1.item())
        test_losses.append(sum(losses) / len(losses))
        test_error.extend(error)

        TP, FP, TN, FN = evaluate_with_confusion(net, test_loader, device, threshold=0.5)
        cm_acc, cm_prec, cm_rec, cm_f1 = confusion_metrics(TP, FP, TN, FN)
        acc.append(cm_acc)
        prec.append(cm_prec)
        rec.append(cm_rec)
        f1.append(cm_f1)
        print(f"[Epoch {epoch}] CM: TP={TP} FP={FP} FN={FN} TN={TN} | "
              f"acc={cm_acc:.3f} prec={cm_prec:.3f} rec={cm_rec:.3f} f1={cm_f1:.3f}")

        if cm_f1 > best_f1:
            best_f1 = cm_f1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)

    weights_dir = Path(config["paths"]["weights"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "CNN_best.pth"
    torch.save(best_state, weights_path)
    print(f"Model saved to {weights_path}")

    # --- Threshold sweep on best checkpoint ---
    net.load_state_dict(best_state)
    all_probs, all_labels = collect_probs(net, test_loader, device)
    best_thresh, thresh_metrics, f1_curve = find_best_threshold(all_probs, all_labels)

    print(f"\n--- Threshold sweep results ---")
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

    os.makedirs('graphs', exist_ok=True)

    # --- Training metrics plot ---
    plt.figure()
    plt.style.use('seaborn-v0_8-muted')
    epoch_range = range(1, EPOCHS + 1)
    plt.plot(epoch_range, acc,  label='Accuracy',  color="#5dcc5d", linewidth=2)
    plt.plot(epoch_range, f1,   label='F1-Score',  color="#f14c4c", linewidth=2)
    plt.plot(epoch_range, prec, label='Precision', color="#41a3db", linewidth=2)
    plt.plot(epoch_range, rec,  label='Recall',    color="#ffa632", linewidth=2)
    plt.title('Model Performance Metrics (threshold=0.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(range(0, EPOCHS + 1, 2))
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join('graphs', 'metrics_graph.png'), dpi=300)
    plt.show()

    # --- Threshold vs F1 plot ---
    thresh_vals = [t for t, _ in f1_curve]
    f1_vals     = [f for _, f in f1_curve]
    plt.figure()
    plt.style.use('seaborn-v0_8-muted')
    plt.plot(thresh_vals, f1_vals, color="#f14c4c", linewidth=2)
    plt.axvline(best_thresh, color="#41a3db", linestyle="--",
                label=f"Best thresh={best_thresh:.2f} (F1={thresh_metrics['f1']:.3f})")
    plt.title('F1 vs Detection Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join('graphs', 'threshold_f1.png'), dpi=300)
    plt.show()

    # --- Confusion matrix at best threshold ---
    TP = thresh_metrics['TP']
    TN = thresh_metrics['TN']
    FP = thresh_metrics['FP']
    FN = thresh_metrics['FN']
    data = np.array([[TN, FP], [FN, TP]])
    labels = np.array([[f'TN\n{TN}', f'FP\n{FP}'],
                       [f'FN\n{FN}', f'TP\n{TP}']])
    cmap = LinearSegmentedColormap.from_list("ambience", ["#f0f0f0", "#fc6b03"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(data, annot=labels, fmt="", cmap=cmap, cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'Confusion Matrix (threshold={best_thresh:.2f})', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('graphs', 'confusion_matrix.png'), dpi=300)
    plt.show()
