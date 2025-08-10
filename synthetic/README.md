# IMU unlearning for Markov chains
The code structure of this project is adapted from the https://github.com/OPTML-Group/Unlearn-Simple.git codebase.

Firstly, please pip install transformers==4.35.2 

### Parameter definition (You can manually adjust the values of these variables)
  ```bash
  state_size=10
  seq_length_retain=20
  seq_length_forget1=20
  seq_length_forget2=20
  num_retain_sequences=10000
  num_forget_sequences1=5000
  num_forget_sequences2=5000
  data_dir="data"
  data_seed=42
  training_seed=42
  unlearning_seed=42
  test_size=0.2
  leakage=0.2
  
  n_embd=128
  n_layer=4
  n_head=4
  activation="softmax"
  ```
### Generate the synthetic data 
  ```bash
  python generate_data.py \
    --state_size $state_size \
    --seq_length_retain $seq_length_retain \
    --seq_length_forget1 $seq_length_forget1 \
    --seq_length_forget2 $seq_length_forget2 \
    --num_retain_sequences $num_retain_sequences \
    --num_forget_sequences1 $num_forget_sequences1 \
    --num_forget_sequences2 $num_forget_sequences2 \
    --data_dir $data_dir \
    --seed $data_seed \
    --test_size $test_size \
    --leakage $leakage
  ```
### Train the original model
```bash
python train.py \
    --state_size $state_size \
    --seq_length_retain $seq_length_retain \
    --seq_length_forget1 $seq_length_forget1 \
    --seq_length_forget2 $seq_length_forget2 \
    --num_retain_sequences $num_retain_sequences \
    --num_forget_sequences1 $num_forget_sequences1 \
    --num_forget_sequences2 $num_forget_sequences2 \
    --data_dir $data_dir \
    --leakage $leakage \
    --n_embd $n_embd \
    --n_layer $n_layer \
    --n_head $n_head \
    --activation $activation \
    --seed $training_seed \
    --batch_size 128 \
    --epochs 5 \
    --learning_rate 0.0005 \
    --model_type pretrain \
    --only_forget1
```
###  Train the retrained mode
```bash
python train.py \
    --state_size $state_size \
    --seq_length_retain $seq_length_retain \
    --seq_length_forget1 $seq_length_forget1 \
    --seq_length_forget2 $seq_length_forget2 \
    --num_retain_sequences $num_retain_sequences \
    --num_forget_sequences1 $num_forget_sequences1 \
    --num_forget_sequences2 $num_forget_sequences2 \
    --data_dir $data_dir \
    --leakage $leakage \
    --n_embd $n_embd \
    --n_layer $n_layer \
    --n_head $n_head \
    --activation $activation \
    --seed $training_seed \
    --batch_size 128 \
    --epochs 5 \
    --learning_rate 0.0005 \
    --model_type retain \
    --only_forget1
```

### Unlearn with IMU method

```bash
python unlearn_IMU.py \
    --state_size $state_size \
    --seq_length_retain $seq_length_retain \
    --seq_length_forget1 $seq_length_forget1 \
    --seq_length_forget2 $seq_length_forget2 \
    --num_retain_sequences $num_retain_sequences \
    --num_forget_sequences1 $num_forget_sequences1 \
    --num_forget_sequences2 $num_forget_sequences2 \
    --data_dir $data_dir \
    --leakage $leakage \
    --n_embd $n_embd \
    --n_layer $n_layer \
    --n_head $n_head \
    --activation $activation \
    --pretraining_batch_size 128 \
    --pretraining_epochs 5 \
    --pretraining_learning_rate 5e-4 \
    --loss_type IMU \
    --seed $unlearning_seed \
    --unlearning_epochs 1 \
    --batch_size 4 \
    --learning_rate 0.0005 \
    --alpha 0.002 \
    --max_iterations 50 \
    --use_retrain_eval
```
### Unlearn with other methods
```bash
python unlearn.py \
    --state_size $state_size \
    --seq_length_retain $seq_length_retain \
    --seq_length_forget1 $seq_length_forget1 \
    --seq_length_forget2 $seq_length_forget2 \
    --num_retain_sequences $num_retain_sequences \
    --num_forget_sequences1 $num_forget_sequences1 \
    --num_forget_sequences2 $num_forget_sequences2 \
    --data_dir $data_dir \
    --leakage $leakage \
    --n_embd $n_embd \
    --n_layer $n_layer \
    --n_head $n_head \
    --activation $activation \
    --pretraining_batch_size 128 \
    --pretraining_epochs 5 \
    --pretraining_learning_rate 5e-4 \
    --loss_type $unelarn_methods \
    --seed $unlearning_seed \
    --unlearning_epochs 1 \
    --batch_size 4 \
    --learning_rate 0.0005 \
    --alpha 0.002 \
    --max_iterations 50 \
    --use_retrain_eval
```

# IMU unlearning for TOFU dataset in LLM scenario
The implementation of the IMU method is provided in the IMU.py, then you can refer to https://github.com/locuslab/open-unlearning.git to add the IMU method and conduct unlearning process.
