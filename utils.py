from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import pandas as pd
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset

preprocess = transforms.Compose(
    [
        lambda img: img.convert("RGB"),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class AnimalDataset(Dataset):
    def __init__(self, df, _type="Species", transform=preprocess):
        self.df = df
        self.type = _type
        self.transform = transform
        self.extra_transform = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        element = self.df.iloc[idx]
        image = Image.open(element["ImageName"]).convert("RGB")
        image = self.transform(image)

        if idx in self.extra_transform:
            image = self.extra_transform[idx](image)

        return image, element[self.type]

    def add_items(self, items, extra_transformations):
        size = len(self.df)
        self.df = pd.concat([self.df, items])
        for i, extra_transformation in enumerate(extra_transformations):
            self.extra_transform[i + size] = extra_transformation


def load_data(data_path="./data/oxford-iiit-pet/images/"):
    # if not, create it
    df = pd.read_csv(
        data_path,
        sep=" ",
        header=None,
        names=["ImageName", "ClassId", "Species", "BreedId"],
        comment="#",
    )
    print("Read the CSV")
    df["ImageName"] = df["ImageName"].apply(
        lambda x: f"data/oxford-iiit-pet/images/{x}.jpg"
    )
    return df


def replace_last_layers(model, layers, is_vit=False, unfreeze_norm=False):
    n_features = model.heads.head.in_features if is_vit else model.fc.in_features
    seq = []
    _in = n_features
    for layer in layers:
        seq.append(torch.nn.Linear(_in, layer))
        _in = layer

    if is_vit:
        model.heads.head = torch.nn.Sequential(*seq)
    else:
        model.fc = torch.nn.Sequential(*seq)

    # Freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    last_layer = model.heads.head if is_vit else model.fc
    for param in last_layer.parameters():
        param.requires_grad = True

    # Unfreeze batch norm layers
    if unfreeze_norm:
        for m in model.modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                ),
            ):
                for param in m.parameters():
                    param.requires_grad = True

    return model


def evaluate_model(model, data_loader, loss_fn, device="cuda:0"):
    model.to(device)
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss += loss_fn(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).sum().item()

    loss /= len(data_loader)
    acc = correct / len(data_loader.dataset)
    return loss, acc


def train(
    model,
    loss_fn,
    optimizer,
    training_loader,
    validation_loader,
    epochs=20,
    store_every_n_batchs=100,
    device="cuda:0",
):
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    steps = []
    step = 0

    model.to(device)
    for epoch in range(epochs):
        for batch in training_loader:
            model.train()
            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()

            if step % store_every_n_batchs == 0:
                total_loss, total_acc = evaluate_model(
                    model, training_loader, loss_fn, device
                )
                train_loss.append(total_loss)
                train_acc.append(total_acc)

                total_loss, total_acc = evaluate_model(
                    model, validation_loader, loss_fn, device
                )
                val_loss.append(total_loss)
                val_acc.append(total_acc)

            step += 1

        print(
            f"Epoch: {epoch+1}/{epochs}.. Train loss: {train_loss[-1]:.3f}.. Train acc: {train_acc[-1]:.3f}.. Val loss: {val_loss[-1]:.3f}.. Val acc: {val_acc[-1]:.3f}",
            end="\r",
        )

    print()
    return (train_loss, train_acc), (val_loss, val_acc)


def test_model(model, test_loader, device="cuda:0"):
    model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            correct += (outputs.argmax(1) == targets).sum().item()

    return correct / len(test_loader.dataset)


def plot_accurcies(train_data, val_data, step_every=100, filename=None):
    train_loss, train_acc = train_data
    val_loss, val_acc = val_data

    steps = [i * step_every for i in range(len(train_loss))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(steps, train_loss, label="Training")
    ax1.plot(steps, val_loss, label="Validation")
    ax1.legend()
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")

    ax2.plot(steps, train_acc, label="Training")
    ax2.plot(steps, val_acc, label="Validation")
    ax2.legend()
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)

    if filename:
        plt.savefig(f"results/{filename}")
    else:
        plt.show()


def store_final_accuracies(train_data, val_data, test_acc, filename=None):
    train_loss, train_acc = train_data
    val_loss, val_acc = val_data

    with open(f"results/{filename}", "w") as f:
        f.write("Loss\n")
        f.write(f"Train: {train_loss[-1]:.3f}\n")
        f.write(f"Valididation: {val_loss[-1]:.3f}\n")
        f.write("\nAccuracy\n")
        f.write(f"Train: {train_acc[-1]:.3f}\n")
        f.write(f"Valididation: {val_acc[-1]:.3f}\n")
        f.write(f"Test: {test_acc:.3f}\n")

    print(
        f"Accuracies:\n - Train: {train_acc[-1]:.3f}\n - Validation: {val_acc[-1]:.3f}\n - Test: {test_acc:.3f}"
    )
