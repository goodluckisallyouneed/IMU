import torch
import torch.nn.functional as F
import torch.nn as nn
from trainer.unlearn.base import UnlearnTrainer

def flatten_grad(grad_list):
    return torch.cat([g.reshape(-1) for g in grad_list])

def get_last_params(model):
    params = [p for n, p in model.named_parameters()
              if any(f"layers.{i}." in n for i in range(19, 22))
              and any(attn in n for attn in ['self_attn.o_proj.weight',
                                             'self_attn.k_proj.weight',
                                             'self_attn.q_proj.weight',
                                             'self_attn.v_proj.weight'])]
    return params

def l1_regularization(model):
    l1_loss = 0.0
    with torch.no_grad():  
        for p in get_last_params(model):
            l1_loss += p.abs().sum()
    return l1_loss

def calc_loss(y, t, state_size):
    return F.cross_entropy(y.view(-1, state_size), t.view(-1), ignore_index=-100)

def grad_z_last(model, x, t, state_size):
    model.eval()
    outputs = model(x)
    y = outputs.logits  
    loss = calc_loss(y, t, state_size)
    last_params = get_last_params(model)
    return torch.autograd.grad(loss, last_params, retain_graph=False)


def compute_fisher_diag_batch(model, x, t, state_size):
    fisher_diag = None
    for i in range(x.size(0)):
        grad_list = grad_z_last(model, x[i].unsqueeze(0), t[i].unsqueeze(0), state_size)
        flat_grad = flatten_grad(grad_list)
        squared_grad = flat_grad ** 2
        fisher_diag = squared_grad if fisher_diag is None else fisher_diag + squared_grad
    fisher_diag /= x.size(0)
    fisher_diag += 0.01
    return 1.0 / fisher_diag  

def compute_influence_for_batch(model, x, t, state_size, device, clamp_quantile=0.93):
    
    inv_fisher_diag = compute_fisher_diag_batch(model, x, t, state_size)

    grads = []
    grad_d2_total = None
    for i in range(x.size(0)):
        grad_list = grad_z_last(model, x[i].unsqueeze(0), t[i].unsqueeze(0), state_size)
        flat_grad = flatten_grad(grad_list)
        grads.append(flat_grad.unsqueeze(0))
        grad_d2_total = flat_grad if grad_d2_total is None else grad_d2_total + flat_grad
    grad_d2_total /= x.size(0)

    G = torch.cat(grads, dim=0)
    weighted = G * inv_fisher_diag.unsqueeze(0)
    influence = -(weighted @ grad_d2_total)  

    influence = torch.where(influence > 0, torch.zeros_like(influence), influence)
    weights = torch.sqrt(torch.abs(influence).float())

    if clamp_quantile is not None and len(weights) > 1:
        max_clip = torch.quantile(weights, clamp_quantile)
        weights = torch.clamp(weights, max=max_clip)

    return weights.to(device)


class IMU(UnlearnTrainer):
    def __init__(self, *args, state_size=None, device="cuda", **kwargs):
        super().__init__(*args, **kwargs)
        self.state_size = state_size
        self.device = device
        if state_size is None and hasattr(self.model, "config"):
            self.state_size = self.model.config.vocab_size
        else:
            self.state_size = state_size
            
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        input_ids = forget_inputs["input_ids"].to(self.device)
        labels = forget_inputs["labels"].to(self.device)

        weights = compute_influence_for_batch(
            model, input_ids, labels, self.state_size, self.device
        )

        outputs = model(**forget_inputs)
        logits = outputs.logits

        ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss_per_token = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss_per_token = loss_per_token.view(labels.size())

        valid_mask = (labels != -100)
        loss_vector = (loss_per_token * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        influence_loss = - torch.dot(weights, loss_vector) / (weights.sum() + 1e-8)
        
        influence_loss = influence_loss + 0.02 * l1_regularization(model)

        def zero_out_grads():
            for p in model.parameters():
                if p not in last_params and p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

        influence_loss.register_hook(lambda grad: (zero_out_grads(), grad)[1])
        
        return (influence_loss, outputs) if return_outputs else influence_loss
