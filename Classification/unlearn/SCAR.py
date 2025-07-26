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
from copy import deepcopy

def accuracy(net, loader, args, single_class=False):
    num_classes = args.num_classes
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_sc = torch.zeros((num_classes))
    correct_sc = torch.zeros((num_classes))

    pred_all = []
    target_all = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if single_class:
            pred_all.append(predicted.detach().cpu())
            target_all.append(targets.detach().cpu())

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
    def __init__(self, net, retain, forget, args, test = None):
        self.net = net
        self.retain = retain
        self.forget = forget
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=args.wd)
        self.epochs = args.scar_epochs
        self.target_accuracy = 0.01
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.scheduler, gamma=0.5)
        self.args = args
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if test is None:
            pass
        else:
            self.test = test

    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        self.net.train()
        for _ in tqdm(range(self.epochs)):
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(inputs, targets)
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

    def evalNet(self):
        self.net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in self.retain:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            correct2 = 0
            total2 = 0
            for inputs, targets in self.forget:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total2 += targets.size(0)
                correct2 += (predicted == targets).sum().item()

            if not(self.test is None):
                correct3 = 0
                total3 = 0
                for inputs, targets in self.test:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total3 += targets.size(0)
                    correct3 += (predicted == targets).sum().item()

        self.net.train()
        if self.test is None:
            return correct / total, correct2 / total2
        else:
            return correct / total, correct2 / total2, correct3 / total3

class RandomLabels(BaseMethod):
    def __init__(self, net, retain, forget,test,class_to_remove=None, args=None):
        super().__init__(net, retain, forget, args, test)
        self.loader = forget
        self.class_to_remove = args.class_to_replace
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.class_to_replace != -1:
            self.random_possible = torch.tensor([i for i in range(args.num_classes) if i != self.class_to_remove]).to(device).to(torch.float32)
        else:
            self.random_possible = torch.tensor([i for i in range(args.num_classes)]).to(device).to(torch.float32)
    def loss_f(self, inputs, targets):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        outputs = self.net(inputs)
        #create a random label tensor of the same shape as the outputs chosing values from self.possible_labels
        random_labels = self.random_possible[torch.randint(low=0, high=self.random_possible.shape[0], size=targets.shape)].to(torch.int64).to(device)
        loss = self.criterion(outputs, random_labels)
        return loss
        


    
class SCAR_model(BaseMethod):
    def __init__(
        self,
        net,
        retain,
        retain_sur,
        forget,
        test,  
        class_to_remove=None,
        args=None
    ):
        super().__init__(net, retain, forget, args, test)
        self.loader = None
        self.class_to_remove = args.class_to_replace
        self.retain_sur = retain_sur
        self.args = args
        self.retain = retain

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
        diff = F.normalize(self.tuckey_transf(samples, delta), p=2, dim=-1)[:, None, :] \
               - F.normalize(mean, p=2, dim=-1)
        right = diff.permute(1, 0, 2) @ S_inv
        return torch.diagonal(right @ diff.permute(1, 2, 0), dim1=1, dim2=2)

    def distill(self, outputs_ret, outputs_original):
        temp = self.args.temperature
        soft_old = F.log_softmax(outputs_original / temp + 1e-5, dim=1)
        soft_new = F.log_softmax(outputs_ret / temp + 1e-5, dim=1)
        return F.kl_div(soft_new, soft_old, reduction='batchmean', log_target=True)

    def tuckey_transf(self, vectors, delta=None):
        if delta is None:
            delta = self.args.delta
        return vectors.pow(delta)

    def run(self):
        # backbone and fc heads
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args = self.args
        epoch = args.scar_epochs
        if args.arch != 'ViT':
            bbone = nn.Sequential(*(list(self.net.children())[:-1] + [nn.Flatten()]))
            fc = self.net.classifier if args.arch == 'AllCNN' else self.net.fc
        else:
            bbone = self.net
            fc = self.net.heads

        original_model = deepcopy(self.net)
        original_model.eval()
        bbone.eval()

        # compute embeddings of retain
        ret_embs, labs = [], []
        with torch.no_grad():
            for img, lab in self.retain:
                img, lab = img.to(device), lab.to(device)
                emb = bbone.forward_encoder(img) if args.arch == 'ViT' else bbone(img)
                ret_embs.append(emb)
                labs.append(lab)
        ret_embs = torch.cat(ret_embs)
        labs = torch.cat(labs)
        print("All retain embeddings shape:", ret_embs.shape)  
        D = ret_embs.shape[1]
        print("Embedding dimension D =", D)                    
        
        distribs, cov_inv = [], []

        unique_labels = torch.unique(labs)
        print(unique_labels)
    
        for i in unique_labels.tolist():
            if isinstance(self.class_to_remove, int) and i == self.class_to_remove:
                continue
        
            samples = self.tuckey_transf(ret_embs[labs == i])
            distribs.append(samples.mean(0))
            cov = torch.cov(samples.T)
            cov2 = self.cov_mat_shrinkage(self.cov_mat_shrinkage(cov))
            cov2 = self.normalize_cov(cov2)
            cov_inv.append(torch.linalg.pinv(cov2))
        distribs = torch.stack(distribs)
        cov_inv = torch.stack(cov_inv)

        bbone.train(); fc.train()
        optimizer = optim.Adam(self.net.parameters(), lr=args.unlearn_lr, weight_decay=args.wd)

        target_acc = 1
        max_retain = 5
        lam1, lam2 = args.lambda_1, args.lambda_2

        all_closest = []
        for epoch in range(args.scar_epochs):
            print(f"SCAR epoch:{epoch}")
            flag_exit = False
            for idx_f, (img_f, lab_f) in enumerate(self.forget):
                for idx_r, batch in enumerate(self.retain_sur):
                    
                    if args.class_to_replace != -1:
                        img_r, lab_r = batch
                    else:
                        img_r, lab_r = batch
                        img_r, lab_r = img_r.to(device), lab_r.to(device)
                        with torch.no_grad():
                          out_orig = original_model(img_r).to(device)
                        out_orig = out_orig.to(device)
    
                    img_r, lab_r = img_r.to(device), lab_r.to(device)
                    img_f = img_f.to(device); lab_f = lab_f.to(device)
                    optimizer.zero_grad()

                    emb_f = bbone.forward_encoder(img_f) if args.arch == 'ViT' else bbone(img_f)
                    dists = self.mahalanobis_dist(emb_f, lab_f, distribs, cov_inv).T
    
                    if idx_r == 0 and epoch == 0:
                        cls = torch.argsort(dists, dim=1)
                        first = cls[:, 0]
                        cls = torch.where(first == lab_f, cls[:, 1], first)
                        all_closest.append(cls)
                    cls = all_closest[idx_f]
                    d = dists[torch.arange(dists.shape[0]), cls[:dists.size(0)]]
            
                    loss_f = d.mean() * lam1
                    out_r = fc(bbone.forward_encoder(img_r)) if args.arch == 'ViT' else fc(bbone(img_r))
                  
                    with torch.no_grad():
                        out_orig = original_model(img_r)
                    
                    if args.class_to_replace != -1:
                        mask = torch.argmax(out_orig, dim=1) != args.class_to_replace
                        out_orig = out_orig[mask]
                        out_r    = out_r[mask]
                   
                    loss_ret = self.distill(out_r, out_orig) * lam2
                    loss = loss_f + loss_ret
    
                    if idx_r > max_retain:
                        break

                    loss.backward()
                    optimizer.step()
    
                    with torch.no_grad():
                        self.net.eval()
                        acc = accuracy(self.net, self.forget, self.args)
                        self.net.train()
                        if acc < target_acc and epoch > 1:
                            flag_exit = True
                    if flag_exit:
                        break
                if flag_exit:
                    break
            if flag_exit:
                    break


            with torch.no_grad():
                acc_f = accuracy(self.net, self.forget, self.args)
                acc_t = accuracy(self.net, self.test, self.args)
                acc_r = accuracy(self.net, self.retain, self.args)
                print(f"Epoch {epoch}: forget acc {acc_f:.3f}, test acc {acc_t:.3f},retain acc {acc_r:.3f}")

        self.net.eval()
        return self.net


        
@iterative_unlearn
def SCAR(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    
    start_time = time.time()
    net_copy = deepcopy(model)
    scar = SCAR_model(
        net=model,
        retain=data_loaders['retain'],
        retain_sur = data_loaders['retain_sur'],
        forget=data_loaders['forget'],
        test=data_loaders['test'],
        class_to_remove=args.class_to_replace,
        args = args
    )

    unlearned_model = scar.run()

    elapsed = time.time() - start_time
    print(f"Epoch {epoch}: SCAR unlearning finished in {elapsed:.2f}s")

    return None

@iterative_unlearn
def Random_l(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    start_time = time.time()
    random_labeler = RandomLabels(
        net=model,
        retain=data_loaders['retain'],
        forget=data_loaders['forget'],
        test=data_loaders['test'],
        class_to_remove=args.class_to_replace,
        args=args
    )

    unlearned_model = random_labeler.run()

    elapsed = time.time() - start_time
    print(f"Epoch {epoch}: Random_Label unlearning finished in {elapsed:.2f}s")

    return unlearned_model
