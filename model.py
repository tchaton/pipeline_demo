# model.py
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

class LitLinearRegression(pl.LightningModule):
    """
    A simple linear regression model.
    This is the core component shared between the training and serving scripts.
    """
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)