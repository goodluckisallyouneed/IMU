import sys
import time

import torch
import torch.nn as nn

import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


class _PGDL2:
    def __init__(self, model, eps=0.04, steps=50, eps_for_division=1e-10):
        self.model = model
        self.eps = float(eps)
        self.steps = int(steps)
        self.eps_for_division = eps_for_division

    @torch.enable_grad()
    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        
        device = images.device
        B = images.size(0)

        eps = torch.full((B,), self.eps, device=device)
        alpha = eps / 10.0

        adv_images_all = images.clone().detach()
        adv_preds_all = torch.full_like(labels, -1)
        active = torch.arange(B, device=device)

        for _outer in range(6):
            if active.numel() == 0:
                break
            cur_imgs = images[active].clone().detach()
            cur_lbls = labels[active]
            adv = cur_imgs.clone()
            cur_eps = eps[active]
            cur_alpha = alpha[active]

            for _ in range(self.steps):
                adv.requires_grad_()
                logits = self.model(adv)
                preds = logits.argmax(1)
                still_correct = preds.eq(cur_lbls)
                adv_preds_all[active[~still_correct]] = preds[~still_correct]
                adv_images_all[active[~still_correct]] = adv.detach()[~still_correct]
                if still_correct.sum() == 0:
                    break

                loss = nn.functional.cross_entropy(logits, cur_lbls)
                grad = torch.autograd.grad(loss, adv)[0]
                grad_norm = (
                    grad.view(grad.size(0), -1).norm(p=2, dim=1)
                    + self.eps_for_division
                )
                grad = grad / grad_norm.view(-1, 1, 1, 1)
                grad[~still_correct] = 0.0

                adv = adv.detach() + cur_alpha.view(-1, 1, 1, 1) * grad
                delta = adv - cur_imgs
                d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
                factor = torch.minimum(
                    cur_eps / d_norm.clamp_min(self.eps_for_division),
                    torch.ones_like(d_norm),
                )
                adv = torch.clamp(
                    cur_imgs + delta * factor.view(-1, 1, 1, 1), 0.0, 1.0
                ).detach()

            with torch.no_grad():
                preds = self.model(adv).argmax(1)
            still_correct = preds.eq(cur_lbls)
            done_mask = ~still_correct
            if done_mask.any():
                adv_preds_all[active[done_mask]] = preds[done_mask]
                adv_images_all[active[done_mask]] = adv[done_mask]
            active = active[still_correct]
            eps[active] = eps[active] * 2.0
            alpha = eps / 10.0

        if (adv_preds_all == -1).any():
            with torch.no_grad():
                logits = self.model(adv_images_all)
                logits.scatter_(1, labels.view(-1, 1), float("-inf"))
                fallback = logits.argmax(1)
            unfilled = adv_preds_all == -1
            adv_preds_all[unfilled] = fallback[unfilled]

        return adv_images_all.detach(), adv_preds_all.detach()


def _fetch_batch(batch, device, imagenet_arch, forget_pid):
    if imagenet_arch:
        return get_x_y_from_data_dict(batch, device)
    if forget_pid is not None:
        x, y = batch["img"], batch["pid"]
    else:
        x, y = batch
    return x.to(device), y.to(device)


def _l1_regularization(model):
    return sum(torch.norm(p, p=1) for p in model.parameters() if p.requires_grad)


def _build_advset(model, forget_loader, args, device):
    imagenet_arch = bool(getattr(args, "imagenet_arch", False))
    forget_pid = getattr(args, "forget_pid", None)
    eps = float(getattr(args, "amun_eps", 0.04))
    steps = int(getattr(args, "amun_steps", 50))

    pgd = _PGDL2(model, eps=eps, steps=steps)
    was_training = model.training
    model.eval()

    adv_xs, adv_ys = [], []
    n_seen = 0
    t0 = time.time()
    for batch in forget_loader:
        image, target = _fetch_batch(batch, device, imagenet_arch, forget_pid)
        adv_x, adv_y = pgd.forward(image, target)
        adv_xs.append(adv_x.detach().cpu())
        adv_ys.append(adv_y.detach().cpu())
        n_seen += image.size(0)
    if was_training:
        model.train()

    adv_xs = torch.cat(adv_xs, dim=0)
    adv_ys = torch.cat(adv_ys, dim=0)
    print(
        f"[AMUN] generated {n_seen} adversarial samples in "
        f"{time.time() - t0:.1f}s (eps={eps}, steps={steps})"
    )
    return torch.utils.data.TensorDataset(adv_xs, adv_ys)


class _DictToTupleDataset(torch.utils.data.Dataset):
   

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if isinstance(sample, dict):
            vals = list(sample.values())
            x, y = vals[0], vals[1]
        else:
            x, y = sample
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        return x, y


def _make_combined_loader(retain_loader, forget_loader, advset, args, use_remain):
    imagenet_arch = bool(getattr(args, "imagenet_arch", False))
    base_loader = retain_loader if use_remain else forget_loader
    base_ds = base_loader.dataset
    if imagenet_arch:
        base_ds = _DictToTupleDataset(base_ds)
    combined = torch.utils.data.ConcatDataset([base_ds, advset])
    return torch.utils.data.DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, "workers", 0),
        pin_memory=True,
    )


@iterative_unlearn
def AMUN(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    use_remain = bool(getattr(args, "amun_use_remain", True))
    amun_alpha_l1 = float(getattr(args, "amun_alpha_l1", 0.0))

    if epoch == 0 or not hasattr(args, "_amun_combined_loader"):
        print("[AMUN] Building adversarial set on the forget loader...")
        advset = _build_advset(model, forget_loader, args, device)
        args._amun_advset = advset
        args._amun_combined_loader = _make_combined_loader(
            retain_loader, forget_loader, advset, args, use_remain
        )
        print(
            f"[AMUN] combined trainset size = "
            f"{len(args._amun_combined_loader.dataset)}  "
            f"(use_remain={use_remain})"
        )

    train_loader = args._amun_combined_loader

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()
    start = time.time()

    for i, batch in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch,
                i + 1,
                optimizer,
                one_epoch_step=len(train_loader),
                args=args,
            )

        
        image, target = batch
        image = image.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()

        output = model(image)
        loss = criterion(output, target)
        if amun_alpha_l1 > 0.0:
            loss = loss + amun_alpha_l1 * _l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()

        if mask is not None:
            for name, param in model.named_parameters():
                if param.grad is not None and name in mask:
                    param.grad = param.grad * mask[name]

        optimizer.step()

        prec1 = utils.accuracy(output.float().data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg
