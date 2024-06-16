import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import pickle
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.ECTEC.seq2seq_model import BartSeq2SeqModel
from src.data.mm_dataset import MMDataset

with open("src/data/dev_data.pkl", "rb") as f:
    dev_data = pickle.load(f)
    dev_data = MMDataset(dev_data)

with open("src/data/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)
    test_data = MMDataset(test_data)

with open("src/data/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)
    train_data = MMDataset(train_data)

train_loader = DataLoader(train_data, batch_size=1)
test_loader = DataLoader(test_data, batch_size=1)
dev_loader = DataLoader(dev_data, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BartSeq2SeqModel()


def save_checkpoint(model, optimizer, criterion, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, criterion, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch


def each_epoch(batch):
    kwargs = batch.copy()
    label = batch["label"]
    kwargs.pop("label")
    # print(kwargs.pop("index"))
    kwargs.pop("index")
    return kwargs, label


def train(num_epochs=20):
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    checkpoint_path = "model_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch + 1, num_epochs):
        # breakpoint()
        total_loss = 0
        model.train()
        for batch in train_loader:
            kwargs, label = each_epoch(batch)
            optimizer.zero_grad()

            outputs = model(**kwargs)
            # breakpoint()
            loss = criterion(outputs.transpose(1, 2), label)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        save_checkpoint(model, optimizer, criterion, epoch, checkpoint_path)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.eval()

        all_predictions = []
        all_targets = []

        for batch in dev_loader:
            kwargs, label = each_epoch(batch)
            outputs = model(**kwargs)

            predictions = torch.argmax(outputs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(label.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        acc = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average="macro")
        recall = recall_score(all_targets, all_predictions, average="macro")
        precision = precision_score(all_targets, all_predictions, average="macro")

        print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")


def test():
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            kwargs, label = each_epoch(batch)

            outputs = model(**kwargs)

            predictions = torch.argmax(outputs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(label.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average="macro")
    recall = recall_score(all_targets, all_predictions, average="macro")
    precision = precision_score(all_targets, all_predictions, average="macro")

    print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")


train()
test()
