# train.py
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# Import the model from our shared file
from model import LitLinearRegression

# --- A simple dataset for our toy problem: y = 2x + 1 + noise ---
class SimpleDataset(Dataset):
    def __init__(self, count=1000):
        self.X = torch.randn(count, 1) * 10
        self.y = 2 * self.X + 1 + torch.randn(count, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    print("--- Starting Training ---")
    
    # 1. Setup data and model
    train_loader = DataLoader(SimpleDataset(), batch_size=32)
    model = LitLinearRegression()

    # 2. Configure the trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto", # Automatically uses GPU/MPS if available
        logger=False,
        enable_checkpointing=False
    )

    # 3. Train the model
    trainer.fit(model, train_loader)

    # 4. Save the final model checkpoint
    pipeline_shared_path = os.getenv('PIPELINE_SHARED_PATH', "")
    checkpoint_path = os.path.join(pipeline_shared_path, "model.ckpt") if pipeline_shared_path else "model.ckpt"

    trainer.save_checkpoint(checkpoint_path)
    
    print(f"\n--- Training Complete. Model saved to {checkpoint_path} ---")
    print("You can now run 'python serve.py' to start the API server.")