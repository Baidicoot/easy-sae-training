import training.dictionary as sae

import json

import numpy as np
import torch
from torchtyping import TensorType
import random
import tqdm

import wandb

import argparse

from utils import dotdict


# can and probably should write a wrapper for state management
# but at that point, what's the difference between this and the old code?
# maybe this is _slightly_ more readable? is that worth it? idk probably not
def train_models(train_cfg: dotdict, dataset_cfg: dict, files_cfg: dotdict, log_cfg: dotdict):
    needs_precision_cast = dataset_cfg["precision"] == "float16"

    activation_size = dataset_cfg["tensor_sizes"][train_cfg.tensor_name]
    latent_dim = activation_size * train_cfg.blowup_ratio

    if train_cfg.l1_penalty_spacing == "log":
        l1_range = list(np.logspace(train_cfg.min_l1_penalty, train_cfg.max_l1_penalty, train_cfg.n_models))
    elif train_cfg.l1_penalty_spacing == "linear":
        l1_range = list(np.linspace(train_cfg.min_l1_penalty, train_cfg.max_l1_penalty, train_cfg.n_models))

    if train_cfg.train_unsparse_baseline:
        l1_range.append(0)

    # l1_range = [0.0001]

    ensemble = sae.make_ensemble(
        activation_size,
        latent_dim,
        l1_range,
        {"lr": train_cfg.adam_lr},
        device=train_cfg.device,
        activation=train_cfg.activation,
    )

    # model = sae.SparseLinearAutoencoder(activation_size, latent_dim, 0.0001, device=train_cfg.device, dtype=torch.float64)

    # optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.adam_lr)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if log_cfg.track_dead_feats is not None:
        activation_counts = torch.zeros(len(l1_range), latent_dim, dtype=torch.long, device=train_cfg.device)
        steps_since_last_check = 0

    for epoch in range(train_cfg.n_epochs):
        print(f"Epoch {epoch}")

        chunk_idxs = np.arange(dataset_cfg["n_chunks"])
        np.random.shuffle(chunk_idxs)

        for chunk in chunk_idxs:
            dataset = torch.load(f"{files_cfg.dataset_folder}/{train_cfg.tensor_name}/{chunk}.pt")

            if needs_precision_cast:
                dataset = dataset.to(torch.float32)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

            for batch in tqdm.tqdm(dataloader):
                batch = batch.to(train_cfg.device)

                loss, mse, hiddens, x_hat, bias_norm, center_norm = ensemble.step_batch(batch)

                if files_cfg.wandb_config is not None:
                    sparsities: TensorType["_n_models"]
                    sparsities = hiddens.abs().gt(train_cfg.nonzero_eps).float().sum(dim=-1).mean(dim=1)

                    if log_cfg.track_dead_feats is not None:
                        activation_counts += hiddens.abs().gt(train_cfg.nonzero_eps).sum(dim=1).long()
                        steps_since_last_check += 1

                        if steps_since_last_check > log_cfg.track_dead_feats:
                            dead_feats = (activation_counts == 0).sum(dim=1)
                            wandb.log(
                                {"dead_feats": {l1: dead_feats[i].item() for i, l1 in enumerate(l1_range)}}, commit=True
                            )
                            activation_counts = torch.zeros_like(activation_counts)
                            steps_since_last_check = 0

                    wandb.log(
                        {
                            "loss": {l1: loss[i].item() for i, l1 in enumerate(l1_range)},
                            "mse": {l1: mse[i].item() for i, l1 in enumerate(l1_range)},
                            "sparsity": {l1: sparsities[i].item() for i, l1 in enumerate(l1_range)},
                            "bias_norm": {l1: bias_norm[i].item() for i, l1 in enumerate(l1_range)},
                            "center_norm": {l1: center_norm[i].item() for i, l1 in enumerate(l1_range)},
                        },
                        commit=True,
                    )

    models = ensemble.unstack()

    model_dict = {model.l1_penalty.item(): model for model in models}

    torch.save(model_dict, files_cfg.save_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, default="activation_data")
    parser.add_argument("--tensor_name", type=str, default="activations")
    parser.add_argument("--blowup_ratio", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--min_l1_penalty", type=float, default=-4)
    parser.add_argument("--max_l1_penalty", type=float, default=-2)
    parser.add_argument("--l1_penalty_spacing", type=str, default="log")
    parser.add_argument("--n_models", type=int, default=10)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--train_unsparse_baseline", action="store_true")
    parser.add_argument("--nonzero_eps", type=float, default=1e-7)
    parser.add_argument("--adam_lr", type=float, default=1e-4)
    parser.add_argument("--save_location", type=str, default="models.pt")
    parser.add_argument("--wandb_config", type=str, default="secrets/wandb_cfg.json")
    parser.add_argument("--track_dead_feats", type=int, default=None)
    parser.add_argument("--load_config", type=str, default=None)
    parser.add_argument("--save_config", type=str, default=None)

    args = parser.parse_args()

    with open(f"{args.dataset_folder}/gen_cfg.json", "r") as f:
        dataset_cfg = json.load(f)

    train_cfg = dotdict(
        {
            "tensor_name": args.tensor_name,
            "blowup_ratio": args.blowup_ratio,
            "device": args.device,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "min_l1_penalty": args.min_l1_penalty,
            "max_l1_penalty": args.max_l1_penalty,
            "l1_penalty_spacing": args.l1_penalty_spacing,
            "n_models": args.n_models,
            "train_unsparse_baseline": args.train_unsparse_baseline,
            "activation": args.activation,
            "nonzero_eps": args.nonzero_eps,
            "adam_lr": args.adam_lr,
        }
    )

    log_cfg = dotdict(
        {
            "track_dead_feats": args.track_dead_feats,
        }
    )

    files_cfg = dotdict(
        {
            "dataset_folder": args.dataset_folder,
            "save_location": args.save_location,
            "wandb_config": args.wandb_config,
            "load_config": args.load_config,
            "save_config": args.save_config,
        }
    )

    if files_cfg.load_config is not None:
        with open(files_cfg.load_config, "r") as f:
            train_cfg = dotdict(json.load(f))

    if files_cfg.save_config is not None:
        with open(files_cfg.save_config, "w") as f:
            json.dump(train_cfg.__dict__, f)

    # initialize wandb
    if files_cfg.wandb_config is not None:
        with open(files_cfg.wandb_config, "r") as f:
            wandb_cfg = json.load(f)
        wandb.login(key=wandb_cfg["api_key"])

        import datetime

        now = datetime.datetime.now()
        timestr = now.strftime("%Y-%m-%d_%H-%M")

        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            config=args.__dict__,
            name=f"{wandb_cfg['run_name']}_{timestr}",
        )

    train_models(train_cfg, dataset_cfg, files_cfg, log_cfg)
