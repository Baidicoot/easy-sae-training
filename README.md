# Easy SAE Training
Easy Sparse Linear Autoencoder (https://arxiv.org/abs/2309.08600) training, with data generated from TransformerLens models. This code represents a simplification of the codebase actually used in the paper, and is designed significantly better and is much more readable. The legacy code is available in the `main` branch.

## Usage

### Installation
Hopefully all the neccesary packages should be in `requirements.txt`.

### Activation Generation
To sample activations, run `generate_test_data.py` with the flags
- `--model [str]` to specify which model to use (using TransformerLens naming)
- `--n_chunks [int]` to specify how many chunks (files containing activations) to generate
- `--chunk_size [int]` to specify the size of the chunks in activations
- `--dataset [str]` to specify the HuggingFace dataset to run on
- `--locations [str]` to specify which activation to sample, using `TransformerLens` hook naming.
- `--dataset_folder [str]` to specify the output folder
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

Trained SAEs are outputted as instances of the `SparseLinearAutoencoder` class (defined in `training/dictionary.py`), as a dictionary indexed by `l1_penalty`.

### Ensembling

Internally, the sweeps over L1 penalty ranges are implemented using a model ensembler defined in `training/ensemble.py`. It should be robust to most modifications of autoencoder architecture, but you might have to fiddle with it if you make strange changes.

### Example Usage

```
python generate_test_data.py --model="EleutherAI/pythia-70m-deduped" --layers 2 --n_chunks=2
python basic_l1_sweep.py --dataset_dir="activation_data/layer_2" --output_dir="output_basic_test" --ratio=8 --batch_size=4096 --n_repetitions=2 --save_after_every
```