from torch.autograd import grad
from tqdm import tqdm
import copy
import os
import time
from collections import OrderedDict
import pickle
from torch.utils.data import Subset
import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate
import numpy as np
import torchreid
from collections import defaultdict
import random

    
def extract_features_only_images(engine, loader):
    features, pids, camids = [], [], []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                imgs, pid, camid = batch[0], batch[1], batch[2]
            else:
                raise ValueError("Unexpected batch format.")

            imgs = imgs.cuda()
            outputs = engine.model(imgs)
            features.append(outputs.cpu())
            pids.extend(pid)
            camids.extend(camid)

    features = torch.cat(features, dim=0)
    return features, np.array(pids), np.array(camids)

def split_reid_by_image(dataset, images_per_pid_val=2, seed=42):
    
    pid_to_indices = defaultdict(list)
    for idx, (_, pid, *_ ) in enumerate(dataset):
        pid_to_indices[pid].append(idx)
    
    random.seed(seed)
    train_indices = []
    val_indices   = []
    for pid, idxs in pid_to_indices.items():
        random.shuffle(idxs)
        val_idxs = idxs[:images_per_pid_val]
        train_idxs = idxs[images_per_pid_val:]
        val_indices.extend(val_idxs)
        train_indices.extend(train_idxs)
    
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    return train_data, val_data
    
def compute_accuracy(loader, model, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                images, targets = batch['img'], batch['pid']
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / total if total > 0 else 0
    
def main():
    start_time = time.time()
    
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )
    #prepare the dataset for reid
    train_manager = torchreid.data.ImageDataManager(
        root='data',       
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100, 
        transforms=['random_flip', 'random_crop']
    )
    
    train_data = train_manager.train_loader.dataset
    train_loader = replace_loader_dataset(train_data, seed=1, shuffle=True)
    
    train_data, val_data = split_reid_by_image(train_data, images_per_pid_val=2)

   
    val_loader = replace_loader_dataset(val_data, seed=1, shuffle=True)
    query_loader, gallery_loader = train_manager.fetch_test_loaders('market1501')


    if args.forget_pid is not None:
        forget_pids = args.forget_pid
        forget_indices = [
            idx
            for idx, sample in enumerate(train_data) if sample['pid'] == args.forget_pid
        ] 
        if len(forget_indices) == 0:
            raise ValueError(f"there are no PID={forget_pids} in the train_set")

    all_indices = set(range(len(train_data)))
    retain_indices = list(all_indices - set(forget_indices))

    forget_data = Subset(train_data, forget_indices)
    retain_data = Subset(train_data, retain_indices)

    forget_loader = replace_loader_dataset(forget_data, seed=1, shuffle=True)
    retain_loader = replace_loader_dataset(retain_data, seed=1, shuffle=True)
    
    assert len(forget_data) + len(retain_data) == len(train_data)

    print(f"number of retain dataset {len(retain_data)}")
    print(f"number of forget dataset {len(forget_data)}")
    
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, query=query_loader, gallery_loader=gallery_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None
    
   
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=train_manager.num_train_pids,
        loss='softmax',
        pretrained=False
    )
    torchreid.utils.load_pretrained_weights(model, args.model_path)
    model = model.to(device)
    original_model = copy.deepcopy(model)
    
    unlearn_method = unlearn.get_unlearn_method(args.unlearn)
    if args.unlearn == "SSD":
        unlearn_method(unlearn_data_loaders, model, criterion, args, train_loader) 
    else:
        unlearn_method(unlearn_data_loaders, model, criterion, args)
    
    unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

   
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total run time: {total_time:.2f} seconds")
    evaluation_result["run time"] = total_time

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )
    
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        train_manager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )
    
    device = next(model.parameters()).device
    

    w1_dist = utils.compute_wasserstein_distance(original_model, model, retain_loader, device, args)
    print(f"First Wasserstein Distance between original & unlearned model (test_loader): {w1_dist:.4f}")
    
    
    print(evaluation_result)

if __name__ == "__main__":
    main()
