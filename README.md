# [Latent State Models of Training Dynamics](https://arxiv.org/abs/2308.09543)

Directly model training dynamics, then interpret the dynamics model.

Check out our runnable [Colab demo!](https://colab.research.google.com/drive/1SFZNNGYwmvRF6TMzrEMk3cAhJh8H0WwN?usp=sharing)

# Usage

All these commands run from the main directory. If you want to run within a directory, you may need to change some imports.

## Step 0: Collect training runs.
You'll need to run these commands several times with different random seeds.

Modular addition, one layer transformer:
```bash
python -m scripts.00_training.modular_addition --output_dir $output_dir --seed $seed 
```

Modular addition, one layer transformer with layer norm:
```bash
python -m scripts.00_training.modular_addition --output_dir $output_dir --seed $seed --use_ln
```

Sparse parities, MLP:
```bash
python -m scripts.00_training.sparse_parities --output_dir $output_dir --seed $seed 
```

MNIST, MLP:
```bash
python -m scripts.00_training.computer_vision --output_dir $output_dir --seed $seed --model_name  mlp --dataset_name mnist
```

CIFAR-100, ResNet18:
```bash
python -m scripts.00_training.computer_vision --output_dir $output_dir --seed $seed --model_name resnet18 --dataset_name cifar100 --use_batch_norm --use_residual
```

CIFAR-100, ResNet18 without batch norm or residual connections:
```bash
python -m scripts.00_training.computer_vision --output_dir $output_dir --seed $seed --model_name resnet18 --dataset_name cifar100
```

## Step 1 (optional): Calculate additional stats.

The training scripts in this repo collect stats automatically, but you may need to run this when using downloaded checkpoints, such as those from [Pythia](https://github.com/EleutherAI/pythia) or [MultiBERTs](https://arxiv.org/abs/2106.16163).
```bash
pythom -m scripts.01_checkpoints_to_stats.compute_stats --model_dir <path/to/downloaded/models> --out_dir $out_dir
```

## Step 2: Collate statistics into 1 file.

Take the stats computed in step 0 or step 1 and organize them into CSVs suitable for training the HMM. `--has_loss` is for cases. `--exp_type` exists to handle the logging of hyperparameters. Valid `exp_type` values include `["modular", "parities", "mnist", "cnn"]` from Step 0.
```bash
python -m scripts.02_stats_to_data.training_run_json_to_csv 
--input_dir <path/to/step/0/files>  --save_dir $save_dir --has_loss --exp_type $exp_type 
```

## Step 3 (optional): Compute additional losses of interest. 

You might use these to try to understand the latent state transitions predicted by the HMM. For example, here we compute values of HANS, an OOD evaluation dataset of natural language inference, to see if there are relationships between the HMM representation of training dynamics and OOD performance.
```bash
python -m scripts.03_compute_losses.hans_eval
```

## Step 4: Train HMM.

Model selection computes the AIC-BIC-log-likelihood curves for varying number of hidden states in the HMM and saves out the best model for each number of hidden states. 
```bash
python -m scripts.04_train_hmm.model_selection \
--data_dir $data_dir --output_file $output_file \
--dataset_name $dataset_name --exp_type $exp_type --cov_type $cov_type --num_iters 32 --max_components 8 
```

Argument glossary: 
- `--dataset_name`: for graph title purposes.
- `--exp_type`: controls the columns of the CSV consumed by the HMM
- `--cov_type`: diagonal or full covariance. If you have few training runs, learning a diagonal covariance matrix is mroe feasible.
- `--max_components`: Too many components is uninterpretable--the HMM becomes as complex as the base model. 8 components is a reasonable cap in our experience.

## Step 5: Analyze the HMM. 

This we do in Jupyter notebooks. Notebooks relevant to this step are labeled with the `05_` tag.
- `05_analyze_state_transitions.ipynb` computes feature movements and important features for state transitions.
- `05_model_selection.ipynb` plots data writen from Step 4 in Seaborn.
- `05_graph_analysis.ipynb` uses linear regression to assign coefficents to latent states (see Section 2.3 in the [paper](https://arxiv.org/abs/2308.09543)) 
- `05_plot.ipynb` creates annotated training figures.

# Citation

Thank you for your interest in our work! If you use this repo, please cite:
```
@article{
    hu2023latent,
    title={Latent State Models of Training Dynamics},
    author={Michael Y. Hu and Angelica Chen and Naomi Saphra and Kyunghyun Cho},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=NE2xXWo0LF},
    note={}
}
```
