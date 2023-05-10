import copy
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from numpy import asarray
import numpy as np
import os
from PIL import Image
from datetime import datetime
import pickle

## Dataset Exploration
datasets.OxfordIIITPet("data", download=True)
DATA_PATH = "./data/oxford-iiit-pet/annotations/list.txt"


def load_data(df_path="./data.pkl"):
    # check if df.csv exists
    if os.path.exists(df_path):
        with open(df_path, "rb") as f:
            return pickle.load(f)

    # if not, create it
    df = pd.read_csv(
        DATA_PATH,
        sep=" ",
        header=None,
        names=["ImageName", "ClassId", "Species", "BreedId"],
        comment="#",
    )
    print(f"Read the CSV")
    df["ImageName"] = df["ImageName"].apply(
        lambda x: f"data/oxford-iiit-pet/images/{x}.jpg"
    )
    print(f"Added the path to the image")
    df["Image"] = df["ImageName"].apply(lambda x: Image.open(x).convert("RGB"))
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    df["Image"] = df["Image"].apply(lambda x: preprocess(x))
    print(f"Preprocessed the images")

    # save df in data as pickle
    with open(df_path, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved the dataframe as a pickle")
    return df


df = load_data()
print(f"The dataset contains {len(df)} images.")

# SPECIES: 1:Cat 2:Dog
# Change to 0:Cat 1:Dog
df["Species"] = df["Species"].apply(lambda x: x - 1)

# BREED ID: 1-25:Cat 1:12:Dog
# Change to 0-24:Cat 25:36:Dog
# check species and then apply transformation
df["BreedId"] = df.apply(
    lambda x: x["BreedId"] - 1 if x["Species"] == 0 else x["BreedId"] + 11, axis=1
)
print(np.sort(df["BreedId"].unique()))

## Load Model

resnet = torch.hub.load(
    "pytorch/vision", "resnet34", weights="ResNet34_Weights.IMAGENET1K_V1"
)
resnet.eval()


def replace_last_layer(model, n_classes):
    # Make a copy of the model
    model = copy.deepcopy(model)

    n_features = model.fc.in_features
    model.fc = torch.nn.Linear(n_features, n_classes)  # new layer

    # Freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True

    return model


model = replace_last_layer(resnet, 2)

## Train Model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Split data into train and test
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["BreedId"]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["BreedId"]
)

print(train_df["Image"].iloc[1].shape)

# raise "stop"


# DataLoader
class AnimalDataset(Dataset):
    def __init__(self, df, _type="Species"):
        self.df = df
        self.type = _type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]["Image"], self.df.iloc[idx][self.type]


training_set = AnimalDataset(train_df)
validation_set = AnimalDataset(val_df)

batch_size = 32
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

# Training
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    loss_fn,
    optimizer,
    training_loader,
    validation_loader,
    epochs=20,
    device="cuda:0",
):
    model.to(device)
    for epoch in range(epochs):
        current_loss = 0.0
        correct = 0

        # Training
        model.train()
        for batch in training_loader:
            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            current_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()

        train_loss = current_loss / len(training_loader)
        train_acc = correct / len(training_loader.dataset)

        val_loss = 0.0
        correct = 0

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                inputs, targets = batch

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                val_loss = loss_fn(outputs, targets).item()
                correct += (outputs.argmax(1) == targets).sum().item()

        val_loss /= len(validation_loader)
        val_acc = correct / len(validation_loader.dataset)

        print(
            f"Epoch: {epoch+1}/{epochs}.. Train loss: {train_loss:.3f}.. Train acc: {train_acc:.3f}.. Val loss: {val_loss:.3f}.. Val acc: {val_acc:.3f}"
        )


train(model, loss, optimizer, train_loader, val_loader, epochs=10, device=device)
