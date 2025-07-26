import torch
import torchreid
import numpy as np
import torchreid.reid.metrics


def extract_features_manual(model, loader, device='cuda'):
    model.eval()
    features, pids, camids, paths = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['img']
            imgs = imgs.to(device)
            outputs = model(imgs) 
            features.append(outputs.cpu())
            batch_pids = batch['pid']
            pids.extend(batch_pids.numpy())
            batch_camids = batch['camid']
            camids.extend(batch_camids.numpy())
    features = torch.cat(features, dim=0).numpy()
    return features, np.asarray(pids), np.asarray(camids)
    
datamanager = torchreid.data.ImageDataManager(
    root='data',       
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,                 
    transforms=['random_flip', 'random_crop']
)

query_loader = datamanager.test_loader['market1501']['query']
gallery_loader = datamanager.test_loader['market1501']['gallery']

model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=False
    )
torchreid.utils.load_pretrained_weights(model, args.unlearned_model)
model = model.cuda()

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
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

qf, q_pids, q_camids = extract_features_manual(model, query_loader)
gf, g_pids, g_camids = extract_features_manual(model, gallery_loader)


distmat = torchreid.reid.metrics.distance.compute_distance_matrix(
    torch.from_numpy(qf), torch.from_numpy(gf), metric='euclidean'
)
cmc, mAP = torchreid.reid.metrics.rank.evaluate_rank(
    distmat.numpy(), q_pids, g_pids, q_camids, g_camids,
    max_rank=50, use_metric_cuhk03=False
)

print(f"mAP: {mAP*100:.2f}%")
print("CMC Top-1~5:", cmc[:5])
