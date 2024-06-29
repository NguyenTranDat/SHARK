import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import os
from dotenv import load_dotenv

from model.mymodel import MyModel
from ulti.ulti import save_model, evalution, load_model, plot_confusion_matrix
from ulti.create_folder import create_folder_if_not_exists, delete_folder_if_exists

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
LOG_CONFUSION_MATRIX = os.getenv("LOG_CONFUSION_MATRIX")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

delete_folder_if_exists(LOG_CONFUSION_MATRIX)
create_folder_if_not_exists(LOG_CONFUSION_MATRIX)


class MintRecModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel().to(device)
        self.criterion = nn.CrossEntropyLoss()

        self.all_labels = []
        self.all_predicted = []

    def forward(self, text, video, audio):
        return self.model(text, video, audio)

    def each_epoch(self, batch, batch_idx):
        index = batch.pop("index")
        target = batch.pop("label")

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
        plot_confusion_matrix(self.all_labels, self.all_predicted, f"{LOG_CONFUSION_MATRIX}/{file_path}")

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

        return optimizer

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.load_state_dict(checkpoint["state_dict"])

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = self.state_dict()
        return checkpoint
