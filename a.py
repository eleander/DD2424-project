import copy
import random
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd
import os
from PIL import Image
import pickle

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

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
    print("Read the CSV")
    df["ImageName"] = df["ImageName"].apply(
        lambda x: f"data/oxford-iiit-pet/images/{x}.jpg"
    )
    print("Added the path to the image")
    df["Image"] = df["ImageName"].apply(lambda x: Image.open(x).convert("RGB"))
    # The images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR,
    # followed by a central crop of crop_size=[224]. Finally the values are first rescaled to [0.0, 1.0] and
    # then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    df["Image"] = df["Image"].apply(lambda x: preprocess(x))
    print("Preprocessed the images")

    # save df in data as pickle
    with open(df_path, "wb") as f:
        pickle.dump(df, f)
    print("Saved the dataframe as a pickle")
    return df


df = load_data()
print(f"The dataset contains {len(df)} images.")

df["Species"] = df["Species"].apply(lambda x: x - 1)
df["ClassId"] = df["ClassId"].apply(lambda x: x - 1)

model = torch.hub.load(
    "pytorch/vision", "vit_b_16", weights="ViT_B_16_Weights.IMAGENET1K_V1"
)


def replace_last_layers(model, layers, unfreeze_norm=False):
    n_features = model.heads.head.in_features
    seq = []
    _in = n_features
    for layer in layers:
        seq.append(torch.nn.Linear(_in, layer))
        _in = layer

    model.heads.head = torch.nn.Sequential(*seq)

    # Freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.head.parameters():
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


class AnimalDataset(Dataset):
    def __init__(self, df, _type="Species"):
        self.df = df
        self.type = _type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]["Image"], self.df.iloc[idx][self.type]


model = replace_last_layers(model, [37], unfreeze_norm=True)

## Train Model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Split data into train and test
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["ClassId"]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["ClassId"]
)

training_set = AnimalDataset(train_df, "ClassId")
validation_set = AnimalDataset(val_df, "ClassId")
test_set = AnimalDataset(test_df, "ClassId")

batch_size = 32


def train(
    model,
    loss_fn,
    optimizer,
    training_loader,
    validation_loader,
    epochs=20,
    device="cuda:0",
):
    train_acc = []
    val_acc = []

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
        train_acc.append(correct / len(training_loader.dataset))

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
        val_acc.append(correct / len(validation_loader.dataset))

        print(
            f"Epoch: {epoch+1}/{epochs}.. Train loss: {train_loss:.3f}.. Train acc: {train_acc[-1]:.3f}.. Val loss: {val_loss:.3f}.. Val acc: {val_acc[-1]:.3f}",
            end="\r",
        )

    print()
    return train_acc, val_acc


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


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

train_acc, val_acc = train(
    model, loss, optimizer, train_loader, val_loader, epochs=10, device="cuda:0"
)
test_acc = test_model(model, test_loader, device="cuda:0")
plot_accurcies(train_acc, val_acc, filename="vit_breed_acc.png")
print(
    f"Accuracies:\n - Train: {train_acc[-1]:.3f}\n - Validation: {val_acc[-1]:.3f}\n - Test: {test_acc:.3f}"
)
