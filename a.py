import random
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
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

model = torch.hub.load(
    "pytorch/vision", "vit_b_16", weights="ViT_B_16_Weights.IMAGENET1K_V1"
)

model = replace_last_layers(model, [37], True, unfreeze_norm=True)

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

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

train_acc, val_acc = train(
    model, loss, optimizer, train_loader, val_loader, epochs=10, device="cuda:0"
)
test_acc = test_model(model, test_loader, device="cuda:0")
filename = "vit_b_16_oxford_iiit_pet"
plot_accurcies(train_acc, val_acc, filename=f"{filename}.png")
store_final_accuracies(train_acc, val_acc, test_acc, filename=f"{filename}_acc.txt")
