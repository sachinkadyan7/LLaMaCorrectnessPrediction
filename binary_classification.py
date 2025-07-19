import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall


class SimpleNNClassifier(pl.LightningModule):
    def __init__(self, input_size=4096, hidden_size=8):
        super(SimpleNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def forward_batch(self, batch):
        x, y = batch
        # Flatten the input
        x = x.view(-1, self.input_size).float()  # Flatten x to shape (batch_size, 4096)
        y = y.view(-1, 1).float()                # Make sure y is of shape (batch_size, 1)

        return self(x), y


    def training_step(self, batch, batch_idx):
        output, y = self.forward_batch(batch)

        loss = nn.BCELoss()(output, y)
        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    
    def val_step(self, batch, _):
        
        output, y = self.forward_batch(batch)
        self.val_accuracy(output, y)
        self.val_f1(output, y)
        self.val_recall(output, y)
        self.val_precision(output, y)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_recall", self.val_recall, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_precision", self.val_precision, prog_bar=True, on_step=True, on_epoch=False)

        return self.val_accuracy
    
    def test_step(self, batch, batch_idx):
        
        output, y = self.forward_batch(batch)
        self.test_accuracy(output, y)
        self.test_f1(output, y)
        self.test_recall(output, y)
        self.test_precision(output, y)
        self.log("test_acc", self.test_accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log("test_f1", self.test_f1, prog_bar=True, on_step=True, on_epoch=False)
        self.log("test_recall", self.test_recall, prog_bar=True, on_step=True, on_epoch=False)
        self.log("test_precision", self.test_precision, prog_bar=True, on_step=True, on_epoch=False)
        
        return self.test_accuracy


    def on_val_epoch_end(self):
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute(), prog_bar=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True)


    def on_test_epoch_end(self):
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)
