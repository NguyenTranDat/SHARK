from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import pickle

from src.lib.ulti import seq_len_to_mask


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


def eval_score(all_targets, all_predictions):
    acc = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average="weighted")
    recall = recall_score(all_targets, all_predictions, average="weighted")
    precision = precision_score(all_targets, all_predictions, average="weighted")

    return acc * 100, f1 * 100, recall * 100, precision * 100


def get_data():
    with open("src/example/log/dev_data.pkl", "rb") as f:
        dev_data = pickle.load(f)

    with open("src/example/log/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    with open("src/example/log/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)

    # with open("src/example/log/mapping2id.pkl", "rb") as f:
    #     mapping2id = pickle.load(f)

    return train_data, dev_data, test_data


def get_loss(tgt_tokens, tgt_seq_len, tgt_emotions, tgt_emo_seq_len, pred):
    tgt_seq_len = tgt_seq_len - 1
    mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
    tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
    loss = F.cross_entropy(target=tgt_tokens, input=pred["pred_ectec"].transpose(1, 2))

    mask_emo = seq_len_to_mask(tgt_emo_seq_len, max_len=tgt_emotions.size(1)).eq(0)
    tgt_emotions = tgt_emotions.masked_fill(mask_emo, -100)
    loss_emo = F.cross_entropy(target=tgt_emotions, input=pred["pred_emo"].transpose(1, 2))
    return loss + loss_emo
