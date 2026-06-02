import torch
import torch.nn.functional as F
import torch.nn as nn
from trainer.unlearn.base import UnlearnTrainer

DOWN_PROJ_KEYS = (
    "mlp.down_proj.weight", 
)

LAYER_RANGE = range(19, 22)


def flatten_grad(grad_list):
    return torch.cat([g.reshape(-1) for g in grad_list])


def _named_head_params(model, layer_range=LAYER_RANGE):
    layer_tags = tuple(f"layers.{i}." for i in layer_range)
    selected = []
    for n, p in model.named_parameters():
        if not any(tag in n for tag in layer_tags):
            continue
        if not any(n.endswith(key) for key in DOWN_PROJ_KEYS):
            continue
        selected.append((n, p))
    return selected


def get_last_params(model, layer_range=LAYER_RANGE):
    named = _named_head_params(model, layer_range)
    if not named:
        # Helpful diagnostic if the architecture name doesn't match.
        sample = sorted({n for n, _ in model.named_parameters()
                         if "mlp" in n})[:8]
        raise RuntimeError(
            "IMU(A1): no MLP down-projection found for layers "
            f"{list(layer_range)} with any of {DOWN_PROJ_KEYS}. "
            f"A few mlp.* parameter names in the model: {sample}"
        )
    return [p for _, p in named]


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
        head_param_ids = {id(p) for p in get_last_params(self.model)}
        for p in self.model.parameters():
            p.requires_grad = id(p) in head_param_ids

        named = _named_head_params(self.model)
        n_head = sum(p.numel() for _, p in named)

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

        return (influence_loss, outputs) if return_outputs else influence_loss
