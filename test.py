import os
from dotenv import load_dotenv

from comet_ml import Experiment
import wandb

# Load environment variables from .env file
load_dotenv(".env")

# Initialize Comet ML
experiment = Experiment(
    api_key=os.getenv("COMET_API_KEY"),
    project_name=os.getenv("COMET_PROJECT_NAME"),
    workspace=os.getenv("COMET_WORKSPACE"),
)

# Initialize WandB
wandb.init(
    project=os.getenv("COMET_PROJECT_NAME"),
    config={
        "learning_rate": 3e-5,
        "batch_size": 8,
        # Add other hyperparameters here
    },
)

# Your code continues here
