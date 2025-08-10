# IMU for Image Classification and Re-ID Task

## IMU for Image Classification
This is the official repository for IMU for image tasks. The code structure of this project is adapted from the [Sparse Unlearn](https://github.com/OPTML-Group/Unlearn-Sparse) codebase.


### Requirements
```bash
pip install -r requirements.txt
```

### Scripts
1. Get the origin model.
    ```bash
    python main_train.py --arch {model name} --dataset {dataset name} --epochs {epochs for training} --lr {learning rate for training} --save_dir {file to save the orgin model}
    ```

2. Unlearn
   
    2.1 Here is the script for forgetting the single class of CIFAR-10 or the entire super-class of CIFAR-100 dataset.
   
    * IMU
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn IMU --class_to_replace ${forgetting class} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --alpha ${alpha}
    ```      

    * Retrain
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn retrain --class_to_replace ${forgetting class} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * FT
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn FT --class_to_replace ${forgetting class} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * GA
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn GA --class_to_replace ${forgetting class} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * IU
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn wfisher --class_to_replace ${forgetting class} --alpha ${alpha}
    ```

    * l1-sparse
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn FT_prune --class_to_replace ${forgetting class} --alpha ${alpha} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

     * SalUn
    ```bash
    python main_random.py --dataset {dataset name} --unlearn RL --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --class_to_replace ${forgetting class} --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path ${saliency_map_path}
    ```

    * SCAR
    ```bash
    python -u main_forget.py --dataset {dataset name} --unlearn SCAR --unlearn_epochs 1 --scar_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --class_to_replace ${forgetting class} --model_path ${origin_model_path} --save_dir ${save_dir} --num_workers 4 --bsize 1024 --temperature ${temperature} --lambda_1 ${lambda_1} --lambda_2 ${lambda_2}
    ```

    * Random_l
    ```bash
    python -u main_forget.py --dataset {dataset name} --unlearn Random_l --unlearn_epochs 1 --scar_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --class_to_replace ${forgetting class} --model_path ${origin_model_path} --save_dir ${save_dir} --num_workers 4
    ```
   * SSD
   ```bash
   python -u main_forget.py --dataset {dataset name} --unlearn SSD --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --class_to_replace ${forgetting class} --ssd_selection_weighting ${ssd_selection_weighting} --ssd_dampening_constant ${ssd_dampening_constant}
   ```
   
  * NPO
  ```bash
  python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn NPO --class_to_replace ${forgetting class} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --beta ${beta}
  ```

  2.2 Here is the script for sample-wise forgetting of CIFAR-10 or CIFAR-100 dataset, changing the number of indexes_to_replace to randomly select samples from dataset and then conduct the unlearning process.
   
  ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn ${unlearn_method} --indexes_to_replace ${number of forgetting samples} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
  ```    

   2.3 Here is the script for forgetting the specific sub-class of CIFAR-100 dataset, which is a more challenging task than entire class forgetting.

 ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset cifar100 --unlearn ${unlearn_method} --class_to_replace ${number of forgetting samples} --type sub_set --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
  ```

   2.4 To explore the unlearn effect of only use the top 50% influencial samples of the forget set in IMU method, here is the example

```bash
  python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --dataset {dataset name} --unlearn IMU --class_to_replace ${forgetting class} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --alpha ${alpha} --top_data 0.5 
```

## IMU for Re-ID task

To begin, please download the market1501 dataset into the data folder.

### Scripts
1. Get the origin model.
    ```bash
   
    pip install torchreid --quiet
    
    python - << 'EOF'
    import torchreid
    
    datamanager = torchreid.data.ImageDataManager(
        root='/kaggle/working',       
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,                 
        transforms=['random_flip', 'random_crop']
    )
    
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )
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
    engine.run(
        save_dir='log/resnet50',
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )
    ```

2. Unlearn
   Here is a example for re-ID unleaning.
   ```bash
   python main_forget_reid.py --seed ${seed} --model_path ${original_model_path} --save_dir ${save_dir} --forget_pid ${target ID to forget} --unlearn ${unlearn_method} --unlearn_epochs ${unlearn_epochs} --unlearn_lr ${unlearn_lr}
   ```
   Note that in the Re-ID task, IMU_REID and SCAR_REID should be used, while all other methods remain unchanged.

3. Evaluate
