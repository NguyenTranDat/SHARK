import os
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW

from src.ECTEC.seq2seq_model import BartSeq2SeqModel
from src.example.mm_dataset import MMDataset
from src.lib.ulti import seq_len_to_mask
from src.ulti import plot_confusion_matrix, save_checkpoint, load_checkpoint, eval_score, get_data, get_loss

BATCH_SIZE = 1
train_data, dev_data, test_data = get_data()

train_data = MMDataset(train_data)
# dev_data = MMDataset(dev_data)
test_data = MMDataset(test_data)

# print(len(train_data))
# print(len(dev_data))
# print(len(test_data))

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)
dev_loader = DataLoader(dev_data, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BartSeq2SeqModel()
model.to(device)


def each_epoch(batch):
    kwargs = batch.copy()
    kwargs.pop("index")
    # print(kwargs.pop("index"))
    return kwargs, kwargs.pop("label")


def check_gradients(model):
    total_gradients = 0
    total_parameters = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            avg_grad = param.grad.abs().mean()
            print(f"Gradient average for {name}: {avg_grad}")
            total_gradients += avg_grad.item()
            total_parameters += 1
    if total_parameters > 0:
        print(f"Average gradient for all parameters: {total_gradients / total_parameters}")


def train(num_epochs=100):
    optimizer = AdamW(model.parameters(), lr=5e-6, correct_bias=False)
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
        total_loss = 0
        model.train()
        for batch in train_loader:
            kwargs, label = each_epoch(batch)
            optimizer.zero_grad()

            outputs = model(**kwargs)

            mask_emo = seq_len_to_mask(torch.LongTensor([label.size(1)]), max_len=label.size(1)).eq(0)
            tgt_emotions = label.masked_fill(mask_emo, -100)
            loss = F.cross_entropy(target=tgt_emotions, input=outputs.transpose(1, 2))

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

        with torch.no_grad():
            for batch in test_loader:
                kwargs, label = each_epoch(batch)
                outputs = model(**kwargs)

                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(label.cpu().numpy())

                # print(predictions, label)

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        acc, f1, recall, precision = eval_score(all_targets, all_predictions)

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

    acc, f1, recall, precision = eval_score(all_targets, all_predictions)

    print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")


train()
# test()
