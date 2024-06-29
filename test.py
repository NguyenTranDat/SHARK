from lightning import Trainer
from process_data.data_module import MIntRecDataModule


mnist = MIntRecDataModule()
mnist.prepare_data()

breakpoint()

trainer = Trainer()

model = LitClassifier()
trainer.fit(model, mnist)
