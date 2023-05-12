import copy
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    plot_accurcies,
    test_model,
    train,
    replace_last_layers,
    load_data,
    store_final_accuracies,
    AnimalDataset,
)

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

## Dataset Exploration
datasets.OxfordIIITPet("data", download=True)
DATA_PATH = "./data/oxford-iiit-pet/annotations/list.txt"

df = load_data(data_path=DATA_PATH)
print(f"The dataset contains {len(df)} images.")

df["Species"] = df["Species"].apply(lambda x: x - 1)
df["ClassId"] = df["ClassId"].apply(lambda x: x - 1)

## Load Model

resnet = torch.hub.load(
    "pytorch/vision", "resnet34", weights="ResNet34_Weights.IMAGENET1K_V1"
)

model = replace_last_layers(copy.deepcopy(resnet), [2])

# Split data into train and test
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["ClassId"]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["ClassId"]
)

training_set = AnimalDataset(train_df)
validation_set = AnimalDataset(val_df)
test_set = AnimalDataset(test_df)

batch_size = 32
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Training
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training")


# FIRST TEST
train_acc, val_acc = train(
    model, loss, optimizer, train_loader, val_loader, epochs=10, device=device
)
test_acc = test_model(model, test_loader, device=device)
filename = "resnet34_oxford_iiit_pet_2"
plot_accurcies(train_acc, val_acc, filename=f"{filename}.png")
store_final_accuracies(
    train_acc[-1], val_acc[-1], test_acc, filename=f"{filename}_acc.txt"
)
print(
    f"Accuracies:\n - Train: {train_acc[-1]:.3f}\n - Validation: {val_acc[-1]:.3f}\n - Test: {test_acc:.3f}"
)


# Augment Train df with (flip, small rotations, crops, small size scaling)
def augment_df(df, p=0.125):
    augmented = []

    possible_transformations = [
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomRotation(45),
        transforms.RandomRotation(135),
        transforms.RandomRotation(225),
        transforms.RandomRotation(315),
    ]

    after_transformation = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
        ]
    )

    for i in range(len(df)):
        data = df.iloc[i].copy()
        img = data["Image"]
        for trans in possible_transformations:
            if random.random() < p:
                img = trans(img)
                img = after_transformation(img)
                data["Image"] = img
                augmented.append(data)

    return pd.concat([df, pd.DataFrame(augmented)])


print(f"Original size: {len(train_df)}")

train_df = augment_df(train_df)
print(f"Augmented size: {len(train_df)}")

training_set = AnimalDataset(train_df, "ClassId")
validation_set = AnimalDataset(val_df, "ClassId")
test_set = AnimalDataset(test_df, "ClassId")

train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

model = replace_last_layers(resnet, [100, 50, 37], unfreeze_norm=True)
loss = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
# Diferent learning rate for sequential layers

# Get parameters of each last sequential layer
layers_params = []
for i, layer in enumerate(model.fc.children()):
    layers_params.append({"params": layer.parameters(), "lr": 1e-3 * 0.1**i})

# Get all other parameters except last sequential layers (model.fc)
other_params = [p for name, p in model.named_parameters() if "fc" not in name]
layers_params.append({"params": other_params, "lr": 1e-4})


optimizer = torch.optim.Adam(layers_params, lr=1e-3, weight_decay=0.001)


# SECOND TEST
train_acc, val_acc = train(
    model, loss, optimizer, train_loader, val_loader, epochs=10, device=device
)
test_acc = test_model(model, test_loader, device="cuda:0")
filename = "resnet34_oxford_iiit_pet_37"
plot_accurcies(train_acc, val_acc, filename=f"{filename}.png")
store_final_accuracies(
    train_acc[-1], val_acc[-1], test_acc, filename=f"{filename}_acc.txt"
)
print(
    f"Accuracies:\n - Train: {train_acc[-1]:.3f}\n - Validation: {val_acc[-1]:.3f}\n - Test: {test_acc:.3f}"
)
