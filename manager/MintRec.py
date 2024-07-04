import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning as L
import os
from dotenv import load_dotenv
import wandb
from comet_ml import Experiment
from datetime import datetime

from model.mymodel import MyModel
from process_data.dataset import MIntRec
from process_data.benchmarks import benchmarks
from ulti.ulti import save_model, evalution, load_model, plot_confusion_matrix
from ulti.create_folder import create_folder_if_not_exists, delete_folder_if_exists
from ulti.read_data import read_pickle
from constants import Config

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb.init(
    project=Config.DATA_VERSION,
    name=f"{Config.DATA_VERSION}_{current_time}",
    config={
        "learning_rate": Config.LEARNING_RATE,
        "architecture": "MULT_SDIF",
        "dataset": Config.DATA_VERSION,
        "epochs": Config.NUM_EPOCH,
    },
)
experiment = Experiment(
    api_key=Config.COMET_API_KEY,
    project_name=Config.DATA_VERSION,
    workspace=Config.COMET_WORKSPACE,
)
experiment.set_name(f"{Config.DATA_VERSION}_{current_time}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benchmark = benchmarks[Config.DATA_VERSION]

# delete_folder_if_exists(Config.LOG_CONFUSION_MATRIX)
# create_folder_if_not_exists(Config.LOG_CONFUSION_MATRIX)


class MintRecModule(L.LightningModule):
    def __init__(self, learning_rate: float = 3e-5):
        super().__init__()
        self.model = MyModel(num_labels=len(benchmark["intent_labels"])).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate

        self.save_hyperparameters()
        wandb.config.update(self.hparams)
        experiment.log_parameters(self.hparams)

        self.all_labels = []
        self.all_predicted = []
        self.loss = []
        self.train_loss = []

    def forward(self, text, video, audio):
        return self.model(text, video, audio)

    def each_epoch(self, batch, batch_idx):
        index = batch.pop("index")
        target = batch.pop("label")
        target_binary = batch.pop("label_binary")

        outputs = self.model(**batch)
        loss = self.criterion(outputs, target)

        return loss, outputs, target

    def training_step(self, batch, batch_idx):
        loss, outputs, target = self.each_epoch(batch, batch_idx)
        self.train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, target = self.each_epoch(batch, batch_idx)
        self.loss.append(loss.item())

        _, predicted = torch.max(outputs, 1)
        self.all_labels.extend(target.cpu().numpy())
        self.all_predicted.extend(predicted.cpu().numpy())

        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, target = self.each_epoch(batch, batch_idx)
        self.loss.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)
        self.all_labels.extend(target.cpu().numpy())
        self.all_predicted.extend(predicted.cpu().numpy())

        return loss

    def on_train_epoch_end(self):
        avg_loss = sum(self.train_loss) / len(self.train_loss)
        wandb.log({"train_loss_epoch": avg_loss})
        experiment.log_metric("train_loss_epoch", avg_loss, epoch=self.current_epoch)
        self.train_loss.clear()

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr})
        experiment.log_metric("learning_rate", current_lr, epoch=self.current_epoch)

    def on_validation_epoch_end(self):
        self.epoch_end(stage="val")

    def on_test_epoch_end(self):
        self.epoch_end(stage="test")

    def epoch_end(self, stage: str):
        accuracy, f1, precision, recall = evalution(self.all_labels, self.all_predicted)
        kwargs_result_metric = {
            f"{stage}_accuracy": accuracy,
            f"{stage}_f1": f1,
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
        }
        wandb.log(kwargs_result_metric)
        experiment.log_metrics(kwargs_result_metric, epoch=self.current_epoch)

        self.all_labels.clear()
        self.all_predicted.clear()

        avg_loss = sum(self.loss) / len(self.loss)
        wandb.log({f"{stage}_loss_epoch": avg_loss})
        experiment.log_metric(f"{stage}_loss_epoch", avg_loss, epoch=self.current_epoch)
        self.loss.clear()

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

        return optimizer

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #         "frequency": 1,
        #     },
        # }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.load_state_dict(checkpoint["state_dict"])

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = self.state_dict()
        return checkpoint

    def argument(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr * 0.5)

        data_argument = read_pickle(f"{Config.LOG_DATA_PATH}/data_argument.pkl")
        data_argument = MIntRec(data_argument)
        data_argument = DataLoader(data_argument, batch_size=Config.BATCH_SIZE, shuffle=True)

        for epoch in range(Config.NUM_EPOCH_ARGUMENT):
            total_loss = 0
            self.model.train()
            for batch in data_argument:
                text = batch.pop("text").to(device)
                target = batch.pop("label").to(device)

                outputs = self.model.argument(text)
                loss = self.criterion(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
