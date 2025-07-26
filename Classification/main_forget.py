from torch.autograd import grad
from tqdm import tqdm
import copy
import os
import time
from collections import OrderedDict
from torchvision.datasets import ImageFolder
import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
import pickle 
import numpy as np
from trainer import validate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.stats import wasserstein_distance

    
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
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()


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

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            if args.dataset == "cifar100" and args.type == "sub_set":            
                def _load_map(root):
                    with open(os.path.join(root, "cifar-100-python", "train"), "rb") as f:
                        d = pickle.load(f, encoding="latin1")
                    m = {}
                    for fl, cl in zip(d["fine_labels"], d["coarse_labels"]):
                        m[fl] = cl
                    return [m[i] for i in range(100)]
                fine2coarse = _load_map(args.data)
                c_id = fine2coarse[args.class_to_replace]
                all_in_coarse = [i for i, c in enumerate(fine2coarse) if c == c_id]
                full_ds = train_loader_full.dataset
                full_targets = np.array(full_ds.targets)
                full_idx = np.where(np.isin(full_targets, all_in_coarse))[0]
                forget_idx = np.where(full_targets == args.class_to_replace)[0]
                retain_idx = np.setdiff1d(full_idx, forget_idx) 
                val_ds = val_loader.dataset
                val_targets = np.array(val_ds.targets)
                val_idx = np.where(np.isin(val_targets, all_in_coarse))[0]
                test_ds = test_loader.dataset
                test_targets = np.array(test_ds.targets)
                test_idx = np.where(np.isin(test_targets, all_in_coarse))[0]
                def make(ds, idxs):
                    new = copy.deepcopy(ds)
                    
                    new.targets = full_targets[idxs]
                    if hasattr(ds, "data"):
                        new.data = ds.data[idxs]
                    return new 
                def makeval(ds, idxs):
                    new = copy.deepcopy(ds)
                    
                    new.targets = val_targets[idxs]
                    if hasattr(ds, "data"):
                        new.data = ds.data[idxs]
                    return new
                def maketest(ds, idxs):
                    new = copy.deepcopy(ds)
                    
                    new.targets = test_targets[idxs]
                    if hasattr(ds, "data"):
                        new.data = ds.data[idxs]
                    return new
                train_loader_full_dataset = make(full_ds, full_idx)
                train_loader_full = replace_loader_dataset(train_loader_full_dataset, seed=seed, shuffle=True)
                forget_dataset = make(full_ds, forget_idx)
                retain_dataset = make(full_ds, retain_idx)
                forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
                retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
                
                val_dataset = makeval(val_ds, val_idx)
                val_loader = replace_loader_dataset(val_dataset, seed=seed, shuffle=False)
                test_dataset = maketest(test_ds, test_idx)
                test_loader = replace_loader_dataset(test_dataset, seed=seed, shuffle=False)
                
                print(f"number of retain dataset {len(retain_dataset)}")
                print(f"number of forget dataset {len(forget_dataset)}")
                print(f"number of full dataset {len(train_loader_full_dataset)}")
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)
        original_model = copy.deepcopy(model)
        unlearn_method = unlearn.get_unlearn_method(args.unlearn)

        if args.unlearn == "SCAR":
            data_path = args.surrogate_dataset_path
            utils.prepare_imagenet_split(data_path)
            surrogate_loader = utils.get_surrogate(args, model)
            unlearn_data_loaders = OrderedDict([
                                        ('retain',     retain_loader),
                                        ('retain_sur', surrogate_loader),
                                        ('forget',     forget_loader),
                                        ('val',        val_loader),
                                        ('test',       test_loader),
                                    ])
            unlearn_method(unlearn_data_loaders, model, criterion, args)
        if args.unlearn == "SSD":
            unlearn_method(unlearn_data_loaders, model, criterion, args, train_loader_full) 
        else:
            unlearn_method(unlearn_data_loaders, model, criterion, args)
            
        unlearn.save_unlearn_checkpoint(model, None, args)


    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )
    
    if evaluation_result is None:
        evaluation_result = {}

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total run time: {total_time:.2f} seconds")
    evaluation_result["run time"] = total_time
    
    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

   
    if args.retrain_model_path:
        retrained_model = copy.deepcopy(model)
        retrain_ckpt = torch.load(args.retrain_model_path, map_location=device)
        if "state_dict" in retrain_ckpt:
            state_dict = retrain_ckpt["state_dict"]
        else:
            state_dict = retrain_ckpt
        retrained_model.load_state_dict(state_dict, strict=False)
        retrained_model.to(device)
        retrained_model.eval()

        
    
        w1_dist = utils.compute_wasserstein_distance(retrained_model, model, test_loader, device, args)
        print(f"First Wasserstein Distance between retrained & unlearned model (test_loader): {w1_dist:.4f}")
    
        evaluation_result["W1_distance_test"] = w1_dist

    
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
