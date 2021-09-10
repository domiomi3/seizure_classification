import wandb
import warnings
from tqdm import tqdm

from src.data_preparation.data_handler import DataHandler
from src.models import ModelMapping
from src.utils.utils import default_argument_parser, set_seed
from src.utils.wandb_logger import setup_wandb
from src.training.train_and_validate import train, validate

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics import Accuracy

warnings.simplefilter("ignore", category=FutureWarning)


def run_experiment(args):
    set_seed(args.random_seed)
    data_handler = DataHandler(data_dir=args.data_dir,
                               batch_size=args.batch_size,
                               data_split=args.data_split,
                               shuffle_dataset=args.shuffle_dataset,
                               weighted_sampler=args.weighted_sampler,
                               model_name=args.model_name,
                               dynamic_window=args.num_models,
                               bandpass_freqs=[args.bandpass_min, args.bandpass_max]
                               )
    train_loader, valid_loader = data_handler.get_data_loaders()

    model = ModelMapping[args.model_name](input_shape=data_handler.get_input_shape(),
                                          num_classes=data_handler.get_num_classes())

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    metric = Accuracy()

    setup_wandb(entity=args.wandb_entity,
                model_name=model.get_name(),
                run_name=args.run_name)

    t = tqdm(range(args.n_epochs))
    for epoch in t:
        train_dict = train(model, train_loader, optimizer, criterion, metric)
        valid_dict = validate(model, valid_loader, criterion, metric)

        wandb.log(train_dict, step=epoch, commit=False)
        wandb.log(valid_dict, step=epoch, commit=False)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    options = vars(args)
    print(options)

    run_experiment(args)
