import argparse
import torch
import random
import numpy as np

from src.models import ModelMapping


def default_argument_parser():

    parser = argparse.ArgumentParser(description="seizure-detection")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Whether to use GPU acceleration."
    )
    parser.add_argument(
        "--data_dir",
        default="data/raw",
        help="Path to TUSZ.",
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=8,
        help="Dynamic window size for the preprocessing step, "
             "also specifies the number of models."
    )
    parser.add_argument(
        "--model_name",
        default="cnn_dense",
        help="Which model to use.",
        choices=ModelMapping.keys()
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--bandpass_min",
        default=0.1,
        type=float,
        help="Min freq for bandpass filtering."
    )
    parser.add_argument(
        "--bandpass_max",
        default=47.0,
        type=float,
        help="Max freq for bandpass filtering."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size."
    )
    parser.add_argument(
        "--data_split",
        default=0.3,
        type=float,
        help="Train/validation split."
    )
    parser.add_argument(
        "--weighted_sampler",
        action="store_true",
        help="Whether to use weighted sampling."
    )
    parser.add_argument(
        "--shuffle_dataset",
        action="store_true",
        help="Whether to shuffle the dataset."
    )
    parser.add_argument(
        "--wandb_entity",
        default="domiomi3",
        help="Wandb account to log experiments into.",
    )
    parser.add_argument(
        "--run_name",
        default="baseline",
        help="Experiment name."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=840,
        help="Random seed."
    )

    return parser


def set_seed(seed):
    """
    Set the seeds for all used libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
