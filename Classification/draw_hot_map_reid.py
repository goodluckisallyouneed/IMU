import os
import arg_parser
import dataset 
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import torchvision
from torchvision.transforms import ToTensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.ResNet import *
import random
import cv2
import torchreid

def main():
    args = arg_parser.parse_args()
    utils.setup_seed(args.seed)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_manager = torchreid.data.ImageDataManager(
        root='data',
        sources='market1501',
        height=256, width=128,
        batch_size_train=32,
        transforms=['random_flip','random_crop']
    )
    train_data = train_manager.train_loader.dataset
    
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=train_manager.num_train_pids,
        loss='softmax',
        pretrained=False
    )
    torchreid.utils.load_pretrained_weights(model, args.model_path)
    model = model.to(device).eval()

    target_layer = model.layer4[-1]  

    cam = GradCAM(model=model, target_layers=[target_layer])

    matching = [i for i, sample in enumerate(train_data) if sample['pid'] == args.forget_pid]

    if len(matching)==0:
        raise ValueError(f'train datasets do not have PID={args.forget_pid}')

    print(f'Found {len(matching)} images for PID={args.forget_pid} in train set.')

    index_show = random.choice(matching)
    sample = train_data[index_show]
    
    img_tensor = sample['img']  
    pid = sample['pid']
    camid = sample['camid']
    
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    
    targets = [ClassifierOutputTarget(int(args.forget_pid))]  
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  
   
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    
    cam_im = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=0.6)

    os.makedirs(args.save_dir, exist_ok=True)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title(f"{args.dataset.upper()} idx={args.forget_pid}")
    plt.axis("off")
    plt.imshow(img_np, interpolation='nearest')

    plt.subplot(1,2,2)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.imshow(cam_im)
    

    plt.tight_layout()
    plt.savefig(args.save_dir, dpi=300)
    print(f"Saved CAM result to {args.save_dir}")
    
if __name__ == "__main__":
    main()
