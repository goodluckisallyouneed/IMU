import time
import torch
import utils
from .impl import iterative_unlearn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, dataset, ConcatDataset
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import copy
import os
import pdb
import math
import shutil
from tqdm import tqdm
import seaborn as sns
import scipy.stats as stats
from typing import Dict, List

class ParameterPerturber:
    def __init__(self, model, optimizer, device, args):
        self.model             = model
        self.opt               = optimizer
        self.device            = device
        self.lower_bound       = args.ssd_lower_bound
        self.exponent          = args.ssd_exponent
        self.magnitude_diff    = args.ssd_magnitude_diff
        self.forget_threshold  = args.ssd_forget_threshold
        self.dampening_constant= args.ssd_dampening_constant
        self.selection_weighting = args.ssd_selection_weighting
        self.min_layer = args.ssd_min_layer
        self.max_layer = args.ssd_max_layer

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def fulllike_params_dict(
        self, model: torch.nn, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        def full_like_tensor(fillval, shape: list) -> list:
            
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=self.device)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary

    def subsample_dataset(self, dataset: dataset, sample_perc: float) -> Subset:
        
        sample_idxs = np.arange(0, len(dataset), step=int((1 / sample_perc)))
        return Subset(dataset, sample_idxs)

    def split_dataset_by_class(self, dataset: dataset) -> List[Subset]:
        
        n_classes = len(set([target for _, target in dataset]))
        subset_idxs = [[] for _ in range(n_classes)]
        for idx, (x, y) in enumerate(dataset):
            subset_idxs[y].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_classes)]

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        
        for batch in dataloader:
            if args.forget_pid is not None:
                x, y = batch['img'], batch['pid']
            else:
                x, y = batch
                
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)


@iterative_unlearn
def SSD(data_loaders, model, criterion, optimizer, epoch, args, train_loader_full, mask=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    perturber = ParameterPerturber(model, optimizer, device, args)
    orig_imp   = perturber.calc_importance(train_loader_full)
    forget_imp = perturber.calc_importance(data_loaders['forget'])
    perturber.modify_weight(orig_imp, forget_imp)
    return 0.0
