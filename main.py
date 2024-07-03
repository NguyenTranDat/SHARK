import os
from dotenv import load_dotenv
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from constants import Config
from process_data.data_module import MIntRecDataModule
from manager.MintRec import MintRecModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_module = MIntRecDataModule()
model = MintRecModule(learning_rate=Config.LEARNING_RATE)
# model.argument()

trainer = L.Trainer(max_epochs=Config.NUM_EPOCH)
trainer.fit(model, datamodule=data_module)

results = trainer.test(model, datamodule=data_module)
