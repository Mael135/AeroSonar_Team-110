import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from aerosonar.models.spectrogramCNN import SpectrogramCNN
from aerosonar.data.dataset import *
from aerosonar.config import load_default_config
LR = 0.00002

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SpectrogramCNN().to(device)
weights = torch.tensor([1.2, 0.8]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(net.parameters(), LR, [0.9,0.99], 1e-10)

config = load_default_config()


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


def evaluate_with_confusion(model, loader, device):
    model.eval()

    TP = FP = TN = FN = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] > 0.3).long()

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




EPOCHS = 20

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3)


train_losses = []
train_acc1 = []
test_losses = []
test_acc1 = []
test_error = []
train_error = []

best_f1 = -1.0
best_state = None

for epoch in range(0, EPOCHS):

    # train for one epoch
    acc1, losses, error = train(train_loader, net, criterion, optimizer, epoch)
    train_acc1.append(acc1.item())
    train_losses.append(sum(losses)/len(losses))
    train_error.extend(error)

    # evaluate on validation set
    acc1, losses, error = validate(test_loader, net, criterion)
    test_acc1.append(acc1.item())
    test_losses.append(sum(losses)/len(losses))
    test_error.extend(error)

    TP, FP, TN, FN = evaluate_with_confusion(net, test_loader, device)
    cm_acc, cm_prec, cm_rec, cm_f1 = confusion_metrics(TP, FP, TN, FN)
    print(f"[Epoch {epoch}] CM: TP={TP} FP={FP} FN={FN} TN={TN} | "
          f"acc={cm_acc:.3f} prec={cm_prec:.3f} rec={cm_rec:.3f} f1={cm_f1:.3f}")
    
    if cm_f1 > best_f1:
        best_f1 = cm_f1
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    # scheduler.step()
    avg_loss = sum(losses) / len(losses)
    scheduler.step(avg_loss)

weights_path = os.path.join(config["paths"]["weights"], "CNN_best.pth")
torch.save(best_state, weights_path)
print("Model saved to drone_model_best.pth")