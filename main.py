import os
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.ECTEC.seq2seq_model import BartSeq2SeqModel
from src.example.mm_dataset import MMDataset
from src.lib.ulti import seq_len_to_mask

with open("src/example/log/train_data.pkl", "rb") as f:
    dev_data = pickle.load(f)
    dev_data = MMDataset(dev_data)

with open("src/example/log/train_data.pkl", "rb") as f:
    test_data = pickle.load(f)
    test_data = MMDataset(test_data)

with open("src/example/log/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)
    train_data = MMDataset(train_data)


with open("src/example/log/mapping2id.pkl", "rb") as f:
    mapping2id = pickle.load(f)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
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

    print("save done!!!!!!")


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


def plot_confusion_matrix(all_targets, all_predictions, file_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(all_targets, all_predictions)
    class_names = [
        "Acknowledge",
        "Advise",
        "Agree",
        "Apologise",
        "Arrange",
        "Ask for help",
        "Asking for opinions",
        "Care",
        "Comfort",
        "Complain",
        "Confirm",
        "Criticize",
        "Doubt",
        "Emphasize",
        "Explain",
        "Flaunt",
        "Greet",
        "Inform",
        "Introduce",
        "Invite",
        "Joke",
        "Leave",
        "Oppose",
        "Plan",
        "Praise",
        "Prevent",
        "Refuse",
        "Taunt",
        "Thank",
        "Warn",
    ]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig(file_path)
    plt.close()


def train(num_epochs=100):
    parameters = []
    params = {"lr": 5e-4, "weight_decay": 1e-2}
    params["params"] = [
        param for name, param in model.named_parameters() if not ("bart_encoder" in name or "bart_decoder" in name)
    ]
    parameters.append(params)

    params = {"lr": 5e-4, "weight_decay": 1e-2}
    params["params"] = []
    for name, param in model.named_parameters():
        if ("bart_encoder" in name or "bart_decoder" in name) and not ("layernorm" in name or "layer_norm" in name):
            params["params"].append(param)
    parameters.append(params)

    params = {"lr": 5e-4, "weight_decay": 0}
    params["params"] = []
    for name, param in model.named_parameters():
        if ("bart_encoder" in name or "bart_decoder" in name) and ("layernorm" in name or "layer_norm" in name):
            params["params"].append(param)
    parameters.append(params)

    optimizer = torch.optim.AdamW(parameters)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    checkpoint_path = "log/model_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, num_epochs):
        # breakpoint()
        total_loss = 0
        model.train()
        for batch in train_loader:
            kwargs, label = each_epoch(batch)
            optimizer.zero_grad()

            outputs = model(**kwargs)
            loss = criterion(outputs.transpose(1, 2), label)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        save_checkpoint(model, optimizer, criterion, epoch, checkpoint_path)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.eval()

        # total_loss = 0
        all_predictions = []
        all_targets = []

        for batch in test_loader:
            kwargs, label = each_epoch(batch)
            outputs = model(**kwargs)

            loss = criterion(outputs.transpose(1, 2), label)
            # total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(label.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        acc = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average="weighted")
        recall = recall_score(all_targets, all_predictions, average="weighted")
        precision = precision_score(all_targets, all_predictions, average="weighted")

        # avg_loss = total_loss / len(test_loader)
        # scheduler.step(avg_loss)

        print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        plot_confusion_matrix(
            all_targets, all_predictions, file_path=f"confusion_matrix/confusion_matrix_epoch_{epoch+1}.png"
        )


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
    f1 = f1_score(all_targets, all_predictions, average="weighted")
    recall = recall_score(all_targets, all_predictions, average="weighted")
    precision = precision_score(all_targets, all_predictions, average="weighted")

    print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")


train()
test()
