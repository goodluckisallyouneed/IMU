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
        if "fc" in name:
            params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

class InfluenceWeightedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, weights):
        self.dataset = base_dataset
        self.weights = weights

    def __getitem__(self, index):
        data, label = self.dataset[index]
        weight = self.weights[index]
        return data, label, weight

    def __len__(self):
        return len(self.dataset)

def compute_fisher_matrix_from_dataset(model, dataset, collate_fn, gpu=-1):
    fisher_diag = None
    count = 0
    for i in tqdm(range(len(dataset)), desc="Compute Diagonal Fisher matrix"):
        x, t = dataset[i]
        x = collate_fn([x])
        t = collate_fn([t])
        grad_list = grad_z_last(x, t, model)
        flat_grad = flatten_grad(grad_list)

        squared_grad = flat_grad ** 2
        if fisher_diag is None:
            fisher_diag = squared_grad
        else:
            fisher_diag += squared_grad
        count += 1

    fisher_diag /= count

    fisher_diag += 0.01
    inv_fisher_diag = 1.0 / fisher_diag

    return inv_fisher_diag.to(torch.device(f"cuda:{gpu}"))

def grad_z_last(z, t, model):
    model.eval()
    z, t = z.cuda(), t.cuda()
    y = model(z)
    loss = calc_loss(y, t)
    last_params = get_last_params(model)
    return torch.autograd.grad(loss, last_params, retain_graph=False)

def get_last_params(model):
    return [p for n, p in model.named_parameters() if "fc" in n]

def flatten_grad(grad_list):
    return torch.cat([g.reshape(-1) for g in grad_list])

def compute_all_gradients(model, dataset, collate_fn):
    model.eval()
    grads = []
    grad_d2_total = None
    for i in tqdm(range(len(dataset)), desc="Compute all sample gradients"):
        x, t = dataset[i]
        x = collate_fn([x])
        t = collate_fn([t])
        grad_list = grad_z_last(x, t, model)
        flat_grad = flatten_grad(grad_list)
        grads.append(flat_grad.unsqueeze(0))
        if grad_d2_total is None:
                grad_d2_total = flat_grad
        else:
            grad_d2_total += flat_grad
    grad_d2_total /= len(dataset)
    return torch.cat(grads, dim=0), grad_d2_total   
  
def compute_influence_weights(model, inv_fisher_diag, dataset, collate_fn):
    
    G, grad_d2_total = compute_all_gradients(model, dataset, collate_fn)  
    weighted_inv_fisher = G * inv_fisher_diag.unsqueeze(0)  
    influences = -(weighted_inv_fisher @ grad_d2_total)  
    return influences.cpu()

def select_top_influential_data(forget_dataset, influences, top_percent):
    inf_arr = np.array(influences)
    sorted_indices = np.argsort(np.abs(inf_arr))[::-1]
    top_k = max(int(len(influences) * top_percent), 1)
    selected_indices = sorted_indices[:top_k]
    print(f"Selecting top {top_percent*100:.1f}% data points, i.e. {top_k} samples.")
    return torch.utils.data.Subset(forget_dataset, selected_indices), selected_indices

def replace_loader_dataset(dataset, args, seed=1, shuffle=True):
    utils.setup_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=shuffle)

def calc_loss(y, t):
    y = torch.nn.functional.log_softmax(y, dim=1)
    return torch.nn.functional.nll_loss(y, t, reduction='mean')

def compute_combined_loss(model, image, target, weight, criterion, args):
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = ('fc' in name)

    loss_vector = criterion(model(image), target)  
    weight = weight.to(loss_vector.device)
    influence_loss = - torch.dot(weight, loss_vector) / weight.sum()
    model.zero_grad()
    #influence_loss.backward(retain_graph=True)
    return influence_loss + args.alpha * l1_regularization(model)

@iterative_unlearn
def IMU(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    criterion = nn.CrossEntropyLoss(reduction="none")
    train_loader = data_loaders["forget"]
    
    if epoch == 0:
        dataset = train_loader.dataset
        collate = train_loader.collate_fn

        print("Computing diagonal Fisher inverse...")
        args.fisher_inv_diag = compute_fisher_matrix_from_dataset(model, dataset, collate, gpu=int(args.gpu))

        print("Computing influence weights for all samples...")
        influences = compute_influence_weights(model, args.fisher_inv_diag, dataset, collate)

        mask = influences < 0
        negative_indices = np.where(mask)[0]
        negative_influences = influences[mask]
        selected_indices = negative_indices
        selected_influences = negative_influences
    
        
        top_percent = args.top_data
        if len(selected_influences) > 0:
            sorted_indices = torch.argsort(torch.abs(selected_influences), descending=True)
            top_k = max(int(len(selected_influences) * top_percent), 1)
            top_indices = sorted_indices[:top_k]
            selected_indices = selected_indices[top_indices]
            selected_influences = selected_influences[top_indices]
    
        selected_subset = torch.utils.data.Subset(dataset, selected_indices)
        weights = torch.sqrt(torch.abs(selected_influences))
        max_clip = torch.quantile(weights, 0.93)
        weights = torch.clamp(weights, max=max_clip)

        
        weighted_dataset = InfluenceWeightedDataset(selected_subset, weights)
        print(f"before the select {len(train_loader.dataset)}")
        train_loader = torch.utils.data.DataLoader(weighted_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        print(f"after the select {len(train_loader.dataset)}")
        args.cached_train_loader = train_loader

    else:
        train_loader = args.cached_train_loader

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()
    start = time.time()
    for i, batch in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)
        
        image, target, weight = batch
        image, target, weight = image.cuda(), target.cuda(), weight.cuda()

        loss = compute_combined_loss(model, image, target, weight, criterion, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = model(image).float()
        prec1 = utils.accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print("Epoch: [{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\tAccuracy {top1.val:.3f} ({top1.avg:.3f})\tTime {3:.2f}".format(
                epoch, i, len(train_loader), end - start, loss=losses, top1=top1))
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg

