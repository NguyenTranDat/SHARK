import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning as L
import os
from dotenv import load_dotenv

from model.mymodel import MyModel
from process_data.dataset import MIntRec
from process_data.benchmarks import benchmarks
from ulti.ulti import save_model, evalution, load_model, plot_confusion_matrix
from ulti.create_folder import create_folder_if_not_exists, delete_folder_if_exists
from ulti.read_data import read_pickle

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

DATA_VERSION = os.getenv("DATA_VERSION")
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
LOG_CONFUSION_MATRIX = os.getenv("LOG_CONFUSION_MATRIX")
NUM_EPOCH_ARGUMENT = int(os.getenv("NUM_EPOCH_ARGUMENT"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
LOG_DATA_PATH = os.getenv("LOG_DATA_PATH")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benchmark = benchmarks[DATA_VERSION]

delete_folder_if_exists(LOG_CONFUSION_MATRIX)
create_folder_if_not_exists(LOG_CONFUSION_MATRIX)


class MintRecModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel(num_labels=len(benchmark["intent_labels"])).to(device)
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        self.all_labels = []
        self.all_predicted = []

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
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, target = self.each_epoch(batch, batch_idx)

        _, predicted = torch.max(outputs, 1)
        self.all_labels.extend(target.cpu().numpy())
        self.all_predicted.extend(predicted.cpu().numpy())

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, target = self.each_epoch(batch, batch_idx)
        self.log("test_loss", loss)

        _, predicted = torch.max(outputs.data, 1)
        self.all_labels.extend(target.cpu().numpy())
        self.all_predicted.extend(predicted.cpu().numpy())

        return loss

    def on_validation_epoch_end(self) -> None:
        self.epoch_end(file_path=f"confusion_matrix_{self.current_epoch}.png")

    def on_test_epoch_end(self) -> None:
        self.epoch_end(file_path=f"confusion_matrix_test.png")

    def epoch_end(self, file_path: str) -> None:
        accuracy, f1, precision, recall = evalution(self.all_labels, self.all_predicted)
        # plot_confusion_matrix(self.all_labels, self.all_predicted, f"{LOG_CONFUSION_MATRIX}/{file_path}")

        self.all_labels.clear()
        self.all_predicted.clear()

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.load_state_dict(checkpoint["state_dict"])

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = self.state_dict()
        return checkpoint

    def argument(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-06, weight_decay=0.1)

        data_argument = read_pickle(f"{LOG_DATA_PATH}/data_argument.pkl")
        data_argument = MIntRec(data_argument)
        data_argument = DataLoader(data_argument, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(NUM_EPOCH_ARGUMENT):
            total_loss = 0
            self.model.train()
            for batch in data_argument:
                text = batch.pop("text").to(device)
                target = batch.pop("label")

                outputs = self.model.argument(text)
                loss = self.criterion(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
