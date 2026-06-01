import math
import sys
import time

import torch
import torch.nn as nn

import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

class MemoryBank:
    
    def __init__(self, size):
        self.grads = []
        self.size = size

    def update(self, grads):
        self.grads.append(grads)
        if len(self.grads) > self.size:
            self.grads.pop(0)

    def mean_grads(self, t_grads):
        grads = []
        for grad in self.grads:
            if torch.cosine_similarity(grad, t_grads, dim=0) < 0:
                grads.append(grad)
        if len(grads) > 0:
            avg_grad = grads[0]
            for grad in grads[1:]:
                avg_grad = avg_grad + grad
            return avg_grad / len(grads)
        return None


def _get_gradient(model: nn.Module):
    gradient = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradient.append(param.grad.detach().clone().view(-1))
        else:
            gradient.append(None)
    return gradient


def _rectify_gradient(grads_x, grads_y):
   
    r_grads_x = []
    r_grads_y = []
    for x, y in zip(grads_x, grads_y):
        if x is None or y is None:
            r_grads_x.append(x)
            r_grads_y.append(y)
            continue
        if torch.cosine_similarity(x, y, dim=0) < 0:
            inp_xy = torch.matmul(y, x)
            inp_xx = torch.norm(x, p=2) ** 2
            inp_yy = torch.norm(y, p=2) ** 2
            x = x - inp_xy / inp_yy * y
            y = y - inp_xy / inp_xx * x  # uses updated x, matching official code
        r_grads_x.append(x)
        r_grads_y.append(y)
    return r_grads_x, r_grads_y


def _fetch_batch(batch, device, imagenet_arch, forget_pid):
    if imagenet_arch:
        image, target = get_x_y_from_data_dict(batch, device)
        return image, target
    if forget_pid is not None:
        image, target = batch["img"], batch["pid"]
    else:
        image, target = batch
    return image.to(device), target.to(device)


@iterative_unlearn
def GDR_GMA(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
   
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    imagenet_arch = bool(getattr(args, "imagenet_arch", False))
    forget_pid = getattr(args, "forget_pid", None)
    gdr_gamma = float(getattr(args, "gdr_gamma", 100.0))
    gdr_epsilon = float(getattr(args, "gdr_epsilon", 0.02))

    if not hasattr(args, "_gdr_bank"):
        forget_size = len(forget_loader.dataset)
        bank_size = max(1, math.ceil(forget_size / max(1, args.batch_size)))
        args._gdr_bank = MemoryBank(size=bank_size)
    bank: MemoryBank = args._gdr_bank

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()
    start = time.time()

    pair_iter = zip(forget_loader, retain_loader)
    for i, (t_batch, n_batch) in enumerate(pair_iter):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch,
                i + 1,
                optimizer,
                one_epoch_step=len(forget_loader),
                args=args,
            )

        t_data, t_labels = _fetch_batch(t_batch, device, imagenet_arch, forget_pid)
        n_data, n_labels = _fetch_batch(n_batch, device, imagenet_arch, forget_pid)

        optimizer.zero_grad()
        n_outputs = model(n_data)
        n_loss = criterion(n_outputs, n_labels)
        n_loss.backward()
        n_grads = _get_gradient(model)

        optimizer.zero_grad()
        t_outputs = model(t_data)
        t_loss = -criterion(t_outputs, t_labels)
        t_loss.backward()
        t_grads = _get_gradient(model)

        if t_grads[-1] is not None:
            bank.update(t_grads[-1])

        r_n_grads, r_t_grads = _rectify_gradient(n_grads, t_grads)
        if epoch > 0 and r_t_grads[-1] is not None:
            mean_g = bank.mean_grads(r_t_grads[-1])
            if mean_g is not None:
                rectified, _ = _rectify_gradient([r_t_grads[-1]], [mean_g])
                r_t_grads[-1] = rectified[-1]

        with torch.no_grad():
            lambda_weight = 1.0 / (
                1.0 + torch.exp(gdr_gamma * (n_loss.detach() - gdr_epsilon))
            )

        optimizer.zero_grad()
        for idx, (_, param) in enumerate(model.named_parameters()):
            if not param.requires_grad:
                continue
            g_n = r_n_grads[idx]
            g_t = r_t_grads[idx]
            if g_n is None or g_t is None:
                continue
            new_grad = (
                (1.0 - lambda_weight) * g_n + lambda_weight * g_t
            ).view(param.size())
            param.grad = new_grad

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None and name in mask:
                    param.grad = param.grad * mask[name]

        optimizer.step()

        output = t_outputs.float().detach()
        prec1 = utils.accuracy(output.data, t_labels)[0]
        losses.update(t_loss.item(), t_data.size(0))
        top1.update(prec1.item(), t_data.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg
