import matplotlib.pyplot as plt
import pandas as pd
import logging
import seaborn as sn
from tabulate import tabulate
import torch
import wandb


class TrainingLogger:
    """Class for logging training results and saving them to wandb.ai if desired.
    Can be used as python logger (e.g. training_logger.error)
    Utilizes python logger with the name 'Training logger'"""

    metrics = {
        'f1-score': lambda x: x.f1_score(),
        'precision': lambda x: x.precision(),
        'recall': lambda x: x.recall(),
        'true-positives': lambda x: x.tp,
        'true-negatives': lambda x: x.tn,
        'false-positives': lambda x: x.fp,
        'false-negatives': lambda x: x.fn
    }

    aggregations = {
        'min': lambda x: torch.min(x, dim=2)[0][0],
        'max': lambda x: torch.max(x, dim=2)[0][0],
        'mean': lambda x: torch.mean(x, dim=2)[0]
    }

    def __init__(self, model_runner, log_weights=False, config={}, log_freq=1):
        """
        Initializes TrainingLogger

        param: model_runner: ModelTrainer of training
        param: log_weights: Flag to enable wandb.ai logging
        param: config: Dictionary of Training settings
        param: log_freq: Frequency of logging (based on logging)
        """
        self.model_runner = model_runner

        self.logger = logging.getLogger("Training logger")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.log_freq = log_freq
        self.log_weights = log_weights
        if log_weights:
            wandb.login()
            if model_runner.runner_type() == "evaluator":
                wandb.init(project=f"{model_runner.task}_evaluation", config=config)
            else:
                wandb.init(project=model_runner.task, config=config)
            wandb.run.name = config["experiment_name"]
        self.metrics_history = {}

    def get_run_id(self):
        """returns wandb.ai run id"""
        if self.log_weights:
            return wandb.run.id
        else:
            return ""

    def setup_model(self, model):
        """
        Call this method on model setup. It will pass the model to wandb.ai if logging is enabled.

        param: model: Model to train
        """
        self.metrics_history = {}
        if self.log_weights:
            wandb.watch(model, log='all', log_freq=self.log_freq)

    def save_model(self, model_path):
        """Saves model to wandb.ai if log_weights is enabled"""
        if self.log_weights:
            wandb.save(model_path)

    def aggregate_metrics(self, phase, metric_name):
        return {aggr_name: aggr_function(self.metrics_history[phase][metric_name])
                for aggr_name, aggr_function in self.aggregations.items()}

    def store_metric_history(self, metric_name, metric_value, phase):
        if not phase in self.metrics_history:
            self.metrics_history[phase] = {}
        if not metric_name in self.metrics_history[phase]:
            self.metrics_history[phase][metric_name] = torch.dstack((metric_value,))
        else:
            self.metrics_history[phase][metric_name] = torch.dstack(
                (self.metrics_history[phase][metric_name], metric_value))

    def print_metrics(self, metrics_object, print_coverage: bool = False):
        """Prints out training or validation metrics.

        param: metrics_object: metrics to print out
        param: print_coverage: Flag to enable coverage printing
        """
        self.logger.info("")

        table_headers = ["metric"]

        id_to_class_mapping = list(self.model_runner.datasets.values())[0].id_to_class_mapping().items()

        for class_id, class_name in id_to_class_mapping:
            table_headers.append(class_name)

        metric_rows = []

        for metric_name, get_method in self.metrics.items():
            model_metric = get_method(metrics_object)
            metric_row = [metric_name]
            for class_id in range(model_metric.shape[0]):
                metric_row.append(model_metric[class_id].item())
            metric_rows.append(metric_row)
            if metric_name == "recall":
                metric_rows.append([])

        self.logger.info(tabulate(metric_rows, headers=table_headers,
                         tablefmt='github', floatfmt=".4f", numalign="center"))

        if print_coverage:
            if isinstance(metrics_object.coverage(), float):
                self.logger.info("Coverage: %f", metrics_object.coverage())
            else:
                for class_id, class_name in id_to_class_mapping:
                    self.logger.info("Coverage for class %s: \t %.4f%%", class_name,
                                     metrics_object.coverage()[class_id].item())

        self.logger.info("")

    def log_metrics(self, model_metrics, phase, epoch, loss=None, log_coverage: bool = False):
        """
        Log metrics from model.

        param: model_metrics: metrics object to print out
        param: phase: Phase - can be train, test or val
        param: epoch: Epoch of the metrics
        param: loss: Set to training loss if available, else None
        param: log_coverage: Flag to enable coverage logging
        """
        self.print_metrics(model_metrics, print_coverage=log_coverage)

        id_to_class_mapping = list(self.model_runner.datasets.values())[0].id_to_class_mapping()

        if self.log_weights and epoch % self.log_freq == 0:
            tolog = {'epoch': epoch}
            if log_coverage:
                if isinstance(model_metrics.coverage(), float):
                    tolog[f"coverage_{phase}"] = model_metrics.coverage()
                else:
                    for class_id, class_name in id_to_class_mapping.items():
                        tolog[f"coverage_{class_name}_{phase}"] = model_metrics.coverage()[class_id]
            if loss:
                tolog["loss"] = loss
            for name, get_method in self.metrics.items():
                metric = get_method(model_metrics)
                if metric.shape.numel() == 1:
                    metric_name = name + '_' + phase
                    tolog[metric_name] = metric
                else:
                    for class_id in range(metric.shape.numel()):
                        class_name = id_to_class_mapping[class_id]
                        metric_name = class_name + '_' + name + '_' + phase
                        tolog[metric_name] = metric[class_id]
            wandb.log(tolog, step=epoch)

    def log_summary(self, best_avg_epoch, best_avg_metrics, best_min_epoch, best_min_metrics, best_epochs_per_class=None, best_metrics_per_class=None):
        """
        Logs training summary. If log_weights is enabled, the summary will be also stored in wandb.ai

        param: best_avg_epoch: Best epoch based on class-average f1-scores
        param: best_avg_metrics: Metrics object of that epoch
        param: best_min_epoch: Best epoch based on minimum f1-score
        param: best_min_metrics: Metrics object of that epoch
        param: best_epochs_per_class: dict of (class_label: best_epoch_for_this_class)
        param: best_metrics_per_class: dict of (class_label: best_metrics_for_this_class)
        """
        self.logger.info("Summary")
        self.logger.info('-' * 10)
        self.logger.info("Validation metrics of model with best average F1-Scores (epoch %s):", best_avg_epoch)
        self.print_metrics(best_avg_metrics)
        self.logger.info("Validation metrics of model with best minimum F1-Score (epoch %s):", best_min_epoch)
        self.print_metrics(best_min_metrics)

        if best_metrics_per_class:
            for class_name, best_class_metrics in best_metrics_per_class.items():
                self.logger.info("Validation metrics of best model for class %s (epoch %s):",
                                 class_name, best_epochs_per_class[class_name])
                self.print_metrics(best_class_metrics)

                if self.log_weights:
                    for metric_name, get_method in self.metrics.items():
                        metric = get_method(best_class_metrics)
                        wandb.run.summary[f"{class_name}_{metric_name}_best_model_val"] = metric[list(
                            self.model_runner.datasets.values())[0].class_name_to_id(class_name)].item()
                        wandb.run.summary[f"{class_name}_best_epoch_val"] = best_epochs_per_class[class_name]

        if self.log_weights:
            for metric_name, get_method in self.metrics.items():
                avg_metric = get_method(best_avg_metrics)
                min_metric = get_method(best_min_metrics)
                for class_id, class_name in list(self.model_runner.datasets.values())[0].id_to_class_mapping().items():
                    wandb.run.summary[f"{class_name}_{metric_name}_avg_model"] = avg_metric[class_id].item()
                    wandb.run.summary[f"{class_name}_best_epoch_avg_model"] = best_avg_epoch
                    wandb.run.summary[f"{class_name}_{metric_name}_min_model"] = min_metric[class_id].item()
                    wandb.run.summary[f"{class_name}_best_epoch_min_model"] = best_min_epoch

    def get(self):
        """Returns the pure python logger for custom usage"""
        return self.logger

    # Workaround to allow calling logging methods on TrainingLogger directly
    def __getattr__(self, name):
        if name in ["debug", "info", "warn", "warning", "error", "exception", "critical"]:
            return getattr(self.logger, name)
        return getattr(self, name)

    def finish(self):
        """Call this method to finish run. If log_weights is enabled, it will close the wandb.ai session"""
        if self.log_weights:
            wandb.run.finish()

    def print_confusion_matrix(self, confusion_matrix: list, title: str):
        """Prints out confusion matrix"""
        class_names = list(self.model_runner.datasets.values())[0].class_names()

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix, index=[i for i in class_names], columns=[i for i in class_names])
        figure = plt.figure(figsize=(10, 7))
        figure.suptitle(title, fontsize=20)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        sn.heatmap(confusion_matrix_df, annot=True)
