import torch
import torchvision
from torch import nn 
from torch import optim
from torch.nn import functional as F
import pickle
from tqdm import tqdm
import time
from copy import deepcopy
import sys
from .impl import iterative_unlearn
from sklearn.decomposition import PCA  

def accuracy(net, loader, args, single_class=False):
    num_classes = args.num_classes
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_sc = torch.zeros((num_classes))
    correct_sc = torch.zeros((num_classes))

    pred_all = []
    target_all = []

    for batch in loader:
        img = batch['img'].to(device)
        lab = batch['pid'].to(device)
        outputs = net(img)
        _, predicted = outputs.max(1)
        total += lab.size(0)
        correct += predicted.eq(lab).sum().item()
        if single_class:
            pred_all.append(predicted.detach().cpu())
            target_all.append(lab.detach().cpu()) 

    if single_class:
        pred_all = torch.cat(pred_all)
        target_all = torch.cat(target_all)
        for i in range(num_classes):
            buff_tar = target_all[target_all==i]
            buff_pred = pred_all[target_all==i]
            total_sc[i] = buff_tar.shape[0]
            correct_sc[i] = (buff_pred == i).sum().item()
        return correct / total, correct_sc/total_sc
    else:
        return correct / total

class BaseMethod:
    def __init__(self, net, retain, forget, args):
        self.net = net
        self.retain = retain
        self.forget = forget
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=args.wd)
        self.epochs = args.scar_epochs
        self.target_accuracy = 0.01
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.scheduler, gamma=0.5)
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        device = self.device
        self.net.train()
        for _ in tqdm(range(self.epochs)):
            for batch in self.retain:
                img = batch['img'].to(device)
                lab = batch['pid'].to(device)
                self.optimizer.zero_grad()
                loss = self.loss_f(img, lab)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.net.eval()
                curr_acc = accuracy(self.net, self.forget, self.args)
                self.net.train()
                print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {self.target_accuracy:.3f}")
                if curr_acc < self.target_accuracy:
                    break
            self.scheduler.step()
        self.net.eval()
        return self.net

class RandomLabels(BaseMethod):
    def __init__(self, net, retain, forget, class_to_remove=None, args=None):
        super().__init__(net, retain, forget, args)
        self.loader = forget
        self.class_to_remove = args.forget_pid
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.forget_pid != -1:
            self.random_possible = torch.tensor([i for i in range(args.num_classes) if i != self.class_to_remove]).to(device).to(torch.float32)
        else:
            self.random_possible = torch.tensor([i for i in range(args.num_classes)]).to(device).to(torch.float32)

    def loss_f(self, inputs, targets):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        outputs = self.net(inputs)
        random_labels = self.random_possible[torch.randint(low=0, high=self.random_possible.shape[0], size=targets.shape)].to(torch.int64).to(device)
        loss = self.criterion(outputs, random_labels)
        return loss

class SCAR_model(BaseMethod):
    def __init__(self, net, retain, forget, class_to_remove=None, args=None):
        super().__init__(net, retain, forget, args)
        self.loader = None
        self.class_to_remove = args.forget_pid
        self.args = args

    def cov_mat_shrinkage(self, cov_mat):
        gamma1 = self.args.gamma1
        gamma2 = self.args.gamma2
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        I = torch.eye(cov_mat.shape[0], device=device)
        V1 = torch.mean(torch.diagonal(cov_mat))
        off_diag = cov_mat.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        V2 = (off_diag * mask).sum() / mask.sum()
        return cov_mat + gamma1 * I * V1 + gamma2 * (1 - I) * V2

    def normalize_cov(self, cov_mat):
        sigma = torch.sqrt(torch.diagonal(cov_mat))
        return cov_mat / (sigma.unsqueeze(1) @ sigma.unsqueeze(0))

    def mahalanobis_dist(self, samples, samples_lab, mean, S_inv):
        delta = self.args.delta
        diff = F.normalize(self.tuckey_transf(samples, delta), p=2, dim=-1)[:, None, :] - F.normalize(mean, p=2, dim=-1)
        right = diff.permute(1, 0, 2) @ S_inv
        return torch.diagonal(right @ diff.permute(1, 2, 0), dim1=1, dim2=2)

    def tuckey_transf(self, vectors, delta=None):
        if delta is None:
            delta = self.args.delta
        return vectors.pow(delta)

    def run(self):
        device = self.device
        args = self.args
        
        pca = PCA(n_components=min(7, 2048))  
      
        if args.arch != 'ViT':
            bbone = nn.Sequential(*(list(self.net.children())[:-1] + [nn.Flatten()]))
            fc = next((layer for name, layer in self.net.named_modules() 
                  if name in ['fc', 'classifier', 'head']), None)
        else:
            bbone = self.net
            fc = self.net.heads

        original_model = deepcopy(self.net)
        original_model.eval()
        bbone.eval()

       
        ret_embs, labs = [], []
        with torch.no_grad():
            for batch in self.retain:
                img = batch['img'].to(device)
                lab = batch['pid'].to(device)
                emb = bbone.forward_encoder(img) if args.arch == 'ViT' else bbone(img)
                ret_embs.append(emb.cpu())  
                labs.append(lab.cpu())
        
        ret_embs = torch.cat(ret_embs).numpy()  
        labs = torch.cat(labs)
        ret_embs = torch.from_numpy(pca.fit_transform(ret_embs)).to(device)  
        
        distribs, cov_inv = [], []
        unique_labels = torch.unique(labs)
        for i in unique_labels.tolist():
            if isinstance(self.class_to_remove, int) and i == self.class_to_remove:
                continue
            samples = self.tuckey_transf(ret_embs[labs == i])
            if len(samples) < 7:
                continue
            distribs.append(samples.mean(0))
            
            
            cov = torch.cov(samples.T)
            cov = self.cov_mat_shrinkage(cov)
            cov = self.normalize_cov(cov)
            cov_inv.append(torch.linalg.pinv(cov).cpu()) 
        
        distribs = torch.stack(distribs).to(device)
        cov_inv = torch.stack(cov_inv).to(device)  

        
        bbone.train()
        fc.train()
        optimizer = optim.Adam(self.net.parameters(), lr=args.unlearn_lr, weight_decay=args.wd)
        
        for epoch in range(args.scar_epochs):
            print(f"SCAR epoch:{epoch}")
            for batch in self.forget:
                img_f = batch['img'].to(device)
                lab_f = batch['pid'].to(device)
                
                optimizer.zero_grad()
                
                
                with torch.no_grad():  
                    emb = bbone(img_f)
                    emb_pca = torch.from_numpy(pca.transform(emb.cpu().detach().numpy())).to(device)
                
                
                emb_pca = emb_pca.clone().requires_grad_(True)  
                
                
                dists = self.mahalanobis_dist(emb_pca, lab_f, distribs, cov_inv).T
                loss = dists.mean() * args.lambda_1
                
               
                print(f"Loss requires grad: {loss.requires_grad}")  
                
                loss.backward()
                optimizer.step()
                
                
                if epoch > 1 and accuracy(self.net, self.forget, self.args) < 0.01:
                    break
        
        self.net.eval()
        return self.net

@iterative_unlearn
def SCAR_REID(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    start_time = time.time()
    scar = SCAR_model(
        net=model,
        retain=data_loaders['retain'],
        forget=data_loaders['forget'],
        class_to_remove=args.forget_pid,
        args=args
    )
    unlearned_model = scar.run()
    elapsed = time.time() - start_time
    print(f"Epoch {epoch}: SCAR unlearning finished in {elapsed:.2f}s")
    return unlearned_model

@iterative_unlearn
def Random_l_REID(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    start_time = time.time()
    random_labeler = RandomLabels(
        net=model,
        retain=data_loaders['retain'],
        forget=data_loaders['forget'],
        class_to_remove=args.forget_pid,
        args=args
    )
    unlearned_model = random_labeler.run()
    elapsed = time.time() - start_time
    print(f"Epoch {epoch}: Random_Label unlearning finished in {elapsed:.2f}s")
  
    return unlearned_model
