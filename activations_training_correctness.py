'''Trains a Classifier to Predict Correctness from LLM Layer Activations'''
from activations_loaders import BatchFileDataset, collate_fn, split_datasets
from torch.utils.data import DataLoader
from binary_classification import SimpleNNClassifier
from layer_classifier import LayerNN
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd
from pathlib import Path


def train_model(cat, layer, task, max_epochs=5):
    output_dir = Path(f".output")
    dataset = BatchFileDataset(output_path=output_dir, cat=cat, layer=layer, task=task)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    train_loader, val_loader, test_loader = split_datasets(data_loader, task=task)

    if task == "correctness":
        model = SimpleNNClassifier(input_size=4096, hidden_size=8).to('cuda:0')
    elif task == "layer":
        model = LayerNN(input_size=4096, hidden_size=8).to('cuda:0')

    checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=1)

    trainer = Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback]) 
    trainer.fit(model, train_loader, val_loader)
    train_metrics = trainer.callback_metrics

    test_result = trainer.test(model, test_loader)


    print(trainer.callback_metrics)
    test_metrics = {
        "dataset": cat,
        "layer": layer,
        "test_acc": test_result[0].get("test_accuracy", None),
        "test_f1": test_result[0].get("test_f1", None),
        "test_precision": test_result[0].get("test_precision", None),
        "test_recall": test_result[0].get("test_recall", None),
    }

    train_metrics.update(test_metrics)
    return train_metrics

def run_correctness_experiment(task="correctness"):
    results = []
    for dataset in ["science_elementary", "arc_hard"]:
        for layer in range(4):
            max_epochs = 5 if task == "correctness" else 50
            metrics = train_model(cat=dataset, layer=layer, task=task, max_epochs=max_epochs)
            
            results.append(metrics)

            if task == "layer":
                break

    # Create DataFrame
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(f"{task}_results.csv", index=False)


if __name__ == "__main__":
    run_correctness_experiment("correctness")