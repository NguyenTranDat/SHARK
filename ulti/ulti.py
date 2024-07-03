import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from process_data.benchmarks import benchmarks
from constants import Config

benchmark = benchmarks[Config.DATA_VERSION]


def save_model(epoch, model, optimizer, criterion, path):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "criterion": criterion.state_dict(),
        },
        path,
    )


def load_model(path, model, optimizer, criterion):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        criterion.load_state_dict(checkpoint["criterion"])

    return model, optimizer, criterion


def evalution(labels, predicts):
    accuracy = 100 * accuracy_score(labels, predicts)
    f1 = 100 * f1_score(labels, predicts, average="weighted", zero_division=0)
    recall = 100 * recall_score(labels, predicts, average="weighted", zero_division=0)
    precision = 100 * precision_score(labels, predicts, average="weighted", zero_division=0)

    print(
        "Test Classification Report:\n",
        classification_report(labels, predicts),
    )

    print(f"Accuracy: {accuracy:.4f}%, F1 Score: {f1:.4f}%, Recall: {recall:.4f}%, Precision: {precision:.4f}%")

    return accuracy, f1, precision, recall


def plot_confusion_matrix(all_targets, all_predictions, file_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(all_targets, all_predictions)
    class_names = benchmark["intent_labels"]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig(file_path)
    plt.close()
