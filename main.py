import os
from dotenv import load_dotenv
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from process_data.data_module import MIntRecDataModule
from manager.MintRec import MintRecModule

load_dotenv(".env")

NUM_EPOCH = int(os.getenv("NUM_EPOCH"))
LOG_MODEL_PATH = os.getenv("LOG_MODEL_PATH")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    dirpath=LOG_MODEL_PATH,  # Directory to save the checkpoints
    filename="mintrec-{epoch:02d}-{val_loss:.2f}",  # Checkpoint filename format
    save_top_k=3,  # Save top 3 models
    mode="min",  # Mode: 'min' for loss, 'max' for accuracy
)

data_module = MIntRecDataModule()
model = MintRecModule()

trainer = L.Trainer(max_epochs=NUM_EPOCH, callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=data_module)
