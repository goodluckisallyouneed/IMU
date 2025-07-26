"""
    setup model and datasets
"""


import copy
import os
import random

# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
import sys
import time
from scipy.stats import wasserstein_distance
import numpy as np
import torch
from dataset import *
from dataset import TinyImageNet
from imagenet import prepare_data
from models import *
from torchvision import transforms

__all__ = [
    "setup_model_dataset",
    "AverageMeter",
    "warmup_lr",
    "save_checkpoint",
    "setup_seed",
    "accuracy",
]


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


def save_checkpoint(
    state, is_SA_best, save_path, pruning, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar")
        )


def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def dataset_convert_to_train(dataset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = train_transform
    dataset.train = False


def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([])
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False


def setup_model_dataset(args):
    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_full_loader, val_loader, _ = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )

        if args.train_seed is None:
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        setup_seed(args.train_seed)

        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "svhn":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052]
        )
        train_full_loader, val_loader, _ = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = svhn_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_full_loader, val_loader, _ = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "TinyImagenet":
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_full_loader, val_loader, test_loader = TinyImageNet(args).data_loaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        # train_full_loader, val_loader, test_loader =None, None,None
        marked_loader, _, _ = TinyImageNet(args).data_loaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
        )
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader

    elif args.dataset == "imagenet":
        classes = 1000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_ys = torch.load(args.train_y_file)
        val_ys = torch.load(args.val_y_file)
        model = model_dict[args.arch](num_classes=classes, imagenet=True)

        model.normalize = normalization
        if args.class_to_replace is None:
            loaders = prepare_data(dataset="imagenet", batch_size=args.batch_size)
            train_loader, val_loader = loaders["train"], loaders["val"]
            return model, train_loader, val_loader
        else:
            train_subset_indices = torch.ones_like(train_ys)
            val_subset_indices = torch.ones_like(val_ys)
            train_subset_indices[train_ys == args.class_to_replace] = 0
            val_subset_indices[val_ys == args.class_to_replace] = 0
            loaders = prepare_data(
                dataset="imagenet",
                batch_size=args.batch_size,
                train_subset_indices=train_subset_indices,
                val_subset_indices=val_subset_indices,
            )
            retain_loader = loaders["train"]
            forget_loader = loaders["fog"]
            val_loader = loaders["val"]
            return model, retain_loader, forget_loader, val_loader

    elif args.dataset == "cifar100_no_val":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set_loader, val_loader, test_loader = cifar100_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    elif args.dataset == "cifar10_no_val":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set_loader, val_loader, test_loader = cifar10_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    else:
        raise ValueError("Dataset not supprot yet !")
    # import pdb;pdb.set_trace()

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    return model, train_set_loader, val_loader, test_loader


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if len(commands) == 0:
        return
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)

    fout = open("stop_{}.sh".format(dir), "w")
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, "w")
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)


def get_loader_from_dataset(dataset, batch_size, seed=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle
    )


def get_unlearn_loader(marked_loader, args):
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = get_loader_from_dataset(
        forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = get_loader_from_dataset(
        retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    print("datasets length: ", len(forget_dataset), len(retain_dataset))
    return forget_loader, retain_loader


def get_poisoned_loader(poison_loader, unpoison_loader, test_loader, poison_func, args):
    poison_dataset = copy.deepcopy(poison_loader.dataset)
    poison_test_dataset = copy.deepcopy(test_loader.dataset)

    poison_dataset.data, poison_dataset.targets = poison_func(
        poison_dataset.data, poison_dataset.targets
    )
    poison_test_dataset.data, poison_test_dataset.targets = poison_func(
        poison_test_dataset.data, poison_test_dataset.targets
    )

    full_dataset = torch.utils.data.ConcatDataset(
        [unpoison_loader.dataset, poison_dataset]
    )

    poisoned_loader = get_loader_from_dataset(
        poison_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    poisoned_full_loader = get_loader_from_dataset(
        full_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    poisoned_test_loader = get_loader_from_dataset(
        poison_test_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )

    return poisoned_loader, unpoison_loader, poisoned_full_loader, poisoned_test_loader

def compute_wasserstein_distance(model1, model2, loader, device, args):
    
    model1.eval()
    model2.eval()

    preds1 = []
    preds2 = []

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            
            out1 = model1(data)
            out2 = model2(data)

            
            _, pred1 = out1.max(dim=1)  
            _, pred2 = out2.max(dim=1)
            preds1 += pred1.cpu().tolist()
            preds2 += pred2.cpu().tolist()

   
    w1_dist = wasserstein_distance(preds1, preds2)
    return w1_dist

class custom_Dset_surrogate(Dataset):
    def __init__(self, dset,labels, logits,transf=None):
        self.dset = dset
        self.labels = labels
        self.logits = logits
        self.transf = transf


    def __len__(self):
        return self.dset.shape[0]

    def __getitem__(self, index):
        x = self.dset[index]
        y = self.labels[index]
        logit_x = self.logits[index]
        if self.transf:
            x=self.transf(x)
        return x, y,logit_x

def get_surrogate(args, original_model=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = '/kaggle/working/Unlearn-Saliency/Classification/scar_re'
    mean = {
        'subset_tiny': (0.485, 0.456, 0.406),
        'subset_Imagenet': (0.4914, 0.4822, 0.4465),
        'subset_rnd_img': (0.5969, 0.5444, 0.4877),
        'subset_COCO': (0.4717,0.4486,0.4089),
        'subset_gaussian_noise': (0,0,0)
    }

    std = {
        'subset_tiny': (0.229, 0.224, 0.225),
        'subset_Imagenet': (0.229, 0.224, 0.225),
        'subset_rnd_img': (0.3366, 0.3260, 0.3411),
        'subset_COCO': (0.2754, 0.2708, 0.2852),
        'subset_gaussian_noise': (1,1,1)
    }

    transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.surrogate_dataset], std[args.surrogate_dataset]),
    ]

    transform_list_test = [
        transforms.ToTensor(),
        transforms.Normalize(mean[args.surrogate_dataset], std[args.surrogate_dataset]),
    ]
    if args.arch == 'ViT':
        transform_list.insert(0, transforms.RandomCrop(224, padding=28))
        transform_list.insert(0, transforms.Resize(224, antialias=True))
        transform_list_test.insert(0, transforms.Resize(224, antialias=True))
    else:
        crop_size = 64 if args.dataset == 'tinyImagenet' else 32
        padding = 8 if args.dataset == 'tinyImagenet' else 4
        transform_list.insert(0, transforms.RandomCrop(crop_size, padding=padding))

    transform_dset = transforms.Compose(transform_list)
    transform_test = transforms.Compose(transform_list_test)

    if args.surrogate_dataset != "subset_gaussian_noise":
        root = os.path.join(data_path, 'surrogate_data/', args.surrogate_dataset + '_split')
        if args.class_to_replace is not None:
            subset = torchvision.datasets.ImageFolder(root=root, transform=transform_dset)
        else:
            subset = torchvision.datasets.ImageFolder(root=root, transform=transform_test)

        if args.surrogate_quantity == -1:
            pass_dataset = subset
        else:
            class_list = list(range(min(args.surrogate_quantity, len(subset.classes))))
            indices = [i for i, (_, lbl) in enumerate(subset.imgs) if lbl in class_list]
            pass_dataset = torch.utils.data.Subset(subset, indices)
    else:
        # Gaussian noise dataset
        if args.surrogate_quantity == -1:
            args.surrogate_quantity = 10
        datasets = []
        for i in range(args.surrogate_quantity):
            fname = f"{data_path}/surrogate_data/{args.surrogate_dataset}_split/{i}/gaussian_noise_{i}.pt"
            imgs = torch.load(fname)
            labels = torch.zeros(imgs.shape[0])
            datasets.append(torch.utils.data.TensorDataset(imgs, labels))
        pass_dataset = torch.utils.data.ConcatDataset(datasets)

    loader_surrogate = DataLoader(
        pass_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    if args.class_to_replace is None:
        # Extract features and logits
        bbone = torch.nn.Sequential(*(list(original_model.children())[:-1] + [torch.nn.Flatten()]))
        fc = original_model.fc
        bbone.eval()

        logits_list, data_list, label_list, feat_list = [], [], [], []
        for img, _ in loader_surrogate:
            img = img.to(device)
            with torch.no_grad():
                out = original_model(img)
                logits_list.append(out.cpu())
                pred_lbl = torch.argmax(out, dim=1).cpu()
                data_list.append(img.cpu())
                label_list.append(pred_lbl)
                feat_list.append(bbone(img).cpu())

        logits = torch.cat(logits_list)
        data_all = torch.cat(data_list)
        labels_all = torch.cat(label_list)
        features_all = torch.cat(feat_list)

        dataset_wlogits = custom_Dset_surrogate(data_all, labels_all, logits)
        print('LEN surrogate', len(dataset_wlogits))

        # compute class sample counts
        class_counts = torch.zeros(args.num_classes, dtype=torch.long)
        for i in range(args.num_classes):
            class_counts[i] = (labels_all == i).sum()
        class_counts[class_counts < 3] = 5
        weights = 1.0 / class_counts.float()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(dataset_wlogits), replacement=True
        )
        loader_surrogate = DataLoader(
            dataset_wlogits,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers
        )

    return loader_surrogate

def prepare_imagenet_split(data_path):
    src = os.path.join(data_path, "subset_Imagenet")
    dst = os.path.join(data_path, "subset_Imagenet_split")
    os.makedirs(dst, exist_ok=True)

    images = glob.glob(os.path.join(src, "*.JPEG"))
    random.shuffle(images)

    for idx, img_file in enumerate(images):
        class_id = idx // 1000
        class_dir = os.path.join(dst, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        os.system(f"cp {img_file} {class_dir}")
