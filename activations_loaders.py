import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

BATCH_SIZE = 32

class BatchFileDataset(Dataset):
    def __init__(self, output_path, cat, layer, task):
        """
        Args:
            output_path (string): Path to the output directory
            cat (string): Category of the dataset (science_elementary, arc_hard)
            layer (int): If task is correctness, this is the layer of the model to use. otherwise, it is ignored
            task (string): Task to perform. Either "correctness" or "layer".
        """
        self.output_path = output_path
        self.x_files_path = f"{output_path}/{cat}/activations"
        self.y_file_path = f"{output_path}/{cat}/{cat}.csv"
        self.batch_size = BATCH_SIZE

        # Each activations is a file with the name {i}_activations.pt. We will load batches one at a time for this. 
        num_x_files = len([f for f in os.listdir(self.x_files_path) if f.endswith('.pt')])
        self.x_files = [f"{i}_activations.pt" for i in range(num_x_files)]

        self.layer = layer
        self.task = task

        # if the task is layer, we have a natural batch of 4 different layers for which we will assign labels 0, 1, 2, 3 at runtime
        # if the task is correctness, we create the batches in advance to match the already batched activations
        self.y_batches = None if not task == "correctness" else self.setup_y_batches()

    def setup_y_batches(self):
        df = pd.read_csv(self.y_file_path)
        
        y = torch.Tensor(df.label).float()

        batches = []
        for idx in range(len(self.x_files)):
            batch = y[idx * self.batch_size:(idx + 1) * self.batch_size]
            batches.append(batch)
        return batches

    def __len__(self):            
        return len(self.x_files)

    def __getitem__(self, idx):
        # Load the x batch file
        x_path = os.path.join(self.x_files_path, self.x_files[idx])
        
        if self.task == "correctness":
            y = self.y_batches[idx]
            x = torch.load(x_path, weights_only=True)[:, self.layer]
        elif self.task == "layer":
            x = torch.load(x_path, weights_only=True)
            # set x device to cuda
            x = x[np.random.randint(0, x.shape[0])].to('cuda:0')
            y = torch.tensor([0., 1., 2., 3.], device='cuda:0', dtype=torch.long)

            idx = torch.randperm(x.shape[0])
            x = x[idx]
            y = y[idx]

        if y.shape[0] != x.shape[0]:
            print(idx, y.shape, x.shape)
        
        return x, y
    
def collate_fn(batch):
    batch = batch[0]
    return batch
    
def get_loader(output_path, cat, layer, task):
    ds = BatchFileDataset(output_path, cat, layer, task)
    return DataLoader(dataset=ds, batch_size=1, collate_fn=collate_fn)
    

def split_datasets(data_loader, task):
    if task == "correctness":
        num_batches = len(data_loader.dataset)
        num_train = int(num_batches * 0.8)
        num_val = int(num_batches * 0.1)
        num_test = num_batches - num_train - num_val
        return torch.utils.data.random_split(data_loader.dataset, [num_train, num_val, num_test])

    elif task == "layer":
        num_batches = len(data_loader.dataset)
        num_train = int(num_batches * 0.8)
        num_val = int(num_batches * 0.1)
        num_test = num_batches - num_train - num_val
        return torch.utils.data.random_split(data_loader.dataset, [num_train, num_val, num_test])
    
