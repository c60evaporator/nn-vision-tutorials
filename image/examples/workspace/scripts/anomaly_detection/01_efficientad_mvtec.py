#%% 
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import EfficientAd
import torch

# Initialize the datamodule, model and engine
datamodule = MVTec(train_batch_size=1, num_workers=0)
model = EfficientAd()
engine = Engine(max_epochs=5)

# Train the model
engine.fit(datamodule=datamodule, model=model)

# %%
