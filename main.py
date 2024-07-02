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
CHECKPOINT_PATH = os.path.join(LOG_MODEL_PATH, "last.ckpt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint callback để lưu model cuối cùng
last_checkpoint_callback = ModelCheckpoint(
    dirpath=LOG_MODEL_PATH,
    filename="mintrec-last",
    save_last=True,
    save_top_k=0,  # Không lưu top k models
)

# Checkpoint callback để lưu model có độ chính xác cao nhất
best_acc_checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",  # Metric để monitor, thay 'val_acc' bằng tên metric của bạn
    dirpath=LOG_MODEL_PATH,
    filename="mintrec-best-{epoch:02d}-{val_acc:.2f}",  # Format tên file
    save_top_k=1,  # Lưu lại model tốt nhất
    mode="max",  # Mode: 'min' cho loss, 'max' cho accuracy
)

data_module = MIntRecDataModule()
model = MintRecModule()

if os.path.exists(CHECKPOINT_PATH):
    last_checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
    current_epoch = last_checkpoint["epoch"]

    print(current_epoch)

    if current_epoch < NUM_EPOCH:
        trainer = L.Trainer(max_epochs=NUM_EPOCH, callbacks=[last_checkpoint_callback, best_acc_checkpoint_callback])
        trainer.fit(model, datamodule=data_module, ckpt_path=CHECKPOINT_PATH)
    else:
        model.load_state_dict(last_checkpoint["state_dict"])
        trainer = L.Trainer()
else:
    model.argument()
    trainer = L.Trainer(max_epochs=NUM_EPOCH, callbacks=[last_checkpoint_callback, best_acc_checkpoint_callback])
    trainer.fit(model, datamodule=data_module)

trainer.test(model, datamodule=data_module)
