# Easy SAE Training
Easy Sparse Linear Autoencoder (https://arxiv.org/abs/2309.08600) training, with data generated from TransformerLens models.

## Usage

### Installation
Hopefully all the neccesary packages should be in `requirements.txt`.

### Activation Generation
To sample activations, run `generate_test_data.py` with the flags
- `--model [str]` to specify which model to use (using TransformerLens naming)
- `--n_chunks [int]` to specify how many chunks (files containing activations) to generate
- `--chunk_size_gb [float]` to specify the size of the chunks in GB
- `--dataset [str]` to specify the HuggingFace dataset to run on
- `--location ['residual'|'mlp'|'attn'|'attn_concat'|'mlpout']` to specify which activation to sample (`'attn_concat'` refers to the attention output before the output linear layer - i.e. concatenated output of heads - and `'mlpout'` refers to the output of the MLP after the output linear layer)
- `--layers [list of int]` to specify which layers to sample at
- `--dataset_folder [str]` to specify the output folder
- `--layer_folder_fmt [str]` to specify a format string for per-layer subfolders
- `--device [str]` to specify a PyTorch device to run the model on

Some of these flags have useful defaults, which you can see in the python file.

### SAE Training
To train an autoencoder, run `basic_l1_sweep.py` with the flags
- `--dataset_dir [str]` to specify the activation dataset
- `--output_dir [str]` to specify where to save the models to
- `--ratio [float]` to specify the 'blowup factor' (features / activation dimensions)
- `--l1_value_min [float]` to specify the minimum L1 penalty factor (log10)
- `--l1_value_max [float]` to specify the maximum L1 penalty factor (log10)
- `--batch_size [int]` to specify the training batch size
- `--device` to specify the PyTorch device to train on
- `--adam_lr` to specify the Adam learning rate
- `--n_repetitions` to specify how many times to train on the dataset
- `--save_after_every` to toggle from saving after every repetion (including the first) to saving after every chunk

Again, some of these flags have useful defaults.

### Output Format

Dictionaries are outputted as instances of the `TiedSAE` class in a list of tuples of hyperparameter settings and the dictionary itself.