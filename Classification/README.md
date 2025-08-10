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
