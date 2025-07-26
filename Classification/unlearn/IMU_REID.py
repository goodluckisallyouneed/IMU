import sys
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import utils
from .impl import iterative_unlearn
import matplotlib.pyplot as plt
sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def l1_regularization(model):
    params_vec = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def compute_fisher_inv_for_last_layer_datapoint(model, data_loader, args):
   
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    fisher_diag = {}
    for name, param in model.named_parameters():
        if "classifier" in name:
            fisher_diag[name] = torch.zeros(param.numel(), device=device)

    total_samples = 0
    
    dataset = data_loader.dataset
    with torch.enable_grad():
        for idx in tqdm(range(len(dataset)), desc="Compute Fisher diagonal"):
            sample = dataset[idx]
            x, y = sample['img'], sample['pid']
            x = x.unsqueeze(0).to(device)
            y = torch.tensor([y], device=device)
    
            output = model(x)
            output = model.classifier(output) 
            loss = criterion(output, y)
            
            params = [p for n,p in model.named_parameters() if "classifier" in n]
            names  = [n for n,p in model.named_parameters() if "classifier" in n]

            grads = grad(loss, 
                         params,
                         retain_graph=False)
            
            for (name, _), g in zip(
                [(n,p) for n,p in model.named_parameters() if "classifier" in n],
                grads
            ):  

                if g is None:
                   g = torch.zeros_like(model.state_dict()[name]).view(-1)
                else:
                   g = g.detach().view(-1)
                fisher_diag[name] += g.pow(2)

            total_samples += 1

    fisher_inv = []
    damping = 1e-2
    for name, diag in fisher_diag.items():
        diag = diag / float(max(total_samples, 1))
        diag = torch.clamp(diag, min=damping)
        fisher_inv.append(diag.reciprocal())
    fisher_inv_flat = torch.cat(fisher_inv)   
    return fisher_inv_flat
   

def compute_batch_influence(model, fisher_inv_diag, image, target, criterion):
    device = image.device
    model.eval()

    for name, p in model.named_parameters():
        p.requires_grad = ("classifier" in name)

    features = model(image)
    output = model.classifier(features)
    loss_vector = criterion(output, target)

    last_params = [p for n, p in model.named_parameters() if "classifier" in n]

    grads = []
    grad_d2_total = None
    for i in range(len(image)):
        model.zero_grad()
        loss_i = loss_vector[i]
        grad_list = torch.autograd.grad(loss_i, last_params, retain_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grad_list])
        grads.append(flat_grad.unsqueeze(0))
        grad_d2_total = flat_grad if grad_d2_total is None else grad_d2_total + flat_grad

    grads = torch.cat(grads, dim=0)
    grad_d2_total /= len(image)

    weighted_inv_fisher = grads * fisher_inv_diag.unsqueeze(0)
    influences = -(weighted_inv_fisher @ grad_d2_total)
    return influences.detach()


def compute_combined_loss(model, image, target, fisher_inv, criterion, args):
    model.train()
    
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name)
    
    loss_vector = criterion(model(image), target)
    influence_weights = compute_batch_influence(model, fisher_inv, image, target, criterion)
    weight = torch.sqrt(influence_weights)
    weight = torch.clamp(weight, max=torch.quantile(weight, 0.95))
    influence_loss = - torch.dot(weight, loss_vector) / weight.sum()
    
    model.zero_grad()
    influence_loss.backward(retain_graph=True)

    return influence_loss + args.alpha * l1_regularization(model)
    
@iterative_unlearn
def IMU_REID(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    criterion = nn.CrossEntropyLoss(reduction="none")
    train_loader = data_loaders["forget"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    all_weights = []
    
    fisher_inv = compute_fisher_inv_for_last_layer_datapoint(model, train_loader, args)
    
    model.train()
    start = time.time()

    for i, batch in enumerate(train_loader):
        image, target = batch['img'], batch['pid']
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.cuda()
        target = target.cuda()

        output_clean = model(image)

        loss = compute_combined_loss(model, image, target, fisher_inv, criterion, args)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        output = output_clean.float()
        prec1 = utils.accuracy(output.data, target)[0]
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
