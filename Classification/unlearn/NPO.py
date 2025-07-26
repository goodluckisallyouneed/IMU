import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import grad
import utils
from .impl import iterative_unlearn
import matplotlib.pyplot as plt
import copy
sys.path.append(".")
from imagenet import get_x_y_from_data_dict

@iterative_unlearn
def NPO(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    train_loader = data_loaders["forget"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
        
    ref_model = copy.deepcopy(model)
    ref_ckpt = torch.load(args.model_path, map_location=device)
    if "state_dict" in ref_ckpt:
        state_dict = ref_ckpt["state_dict"]
    else:
        state_dict = ref_ckpt
        
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.cuda()

    model.train()
    start = time.time()

    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )
    
        image = image.cuda()
        target = target.cuda()
        
        output = model(image)               
        with torch.no_grad():
            ref_output = ref_model(image)   

        prob = torch.softmax(output, dim=1)
        prob_ref = torch.softmax(ref_output, dim=1)

        y_idx = target.view(-1, 1)
        pi_theta = prob.gather(1, y_idx).squeeze(1)         
        pi_ref = prob_ref.gather(1, y_idx).squeeze(1)

        beta = args.beta
        w = 2 * (pi_theta ** beta) / ((pi_theta ** beta) + (pi_ref ** beta) + 1e-8)  

        per_sample_loss = -torch.log(pi_theta + 1e-8)      
        weighted_loss = (w * per_sample_loss).mean()

        optimizer.zero_grad()
        weighted_loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        prec1 = utils.accuracy(output.data, target)[0]
        losses.update(weighted_loss.item(), image.size(0))
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
