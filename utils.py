from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import pandas as pd
import os
import pickle
from PIL import Image

preprocess = transforms.Compose(
    [
        lambda img: img.convert("RGB"),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_data(df_path="./data.pkl", data_path="./data/oxford-iiit-pet/images/"):
    # check if df.csv exists
    if os.path.exists(df_path):
        with open(df_path, "rb") as f:
            return pickle.load(f)

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


def plot_accurcies(train_acc, val_acc, filename=None):
    plt.plot(train_acc, label="Training")
    plt.plot(val_acc, label="Validation")
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    if filename:
        plt.savefig(f"results/{filename}")
    else:
        plt.show()


def store_final_accuracies(train_acc, val_acc, test_acc, filename=None):
    with open(f"results/{filename}", "w") as f:
        f.write(f"Train: {train_acc:.3f}\n")
        f.write(f"Val: {val_acc:.3f}\n")
        f.write(f"Test: {test_acc:.3f}\n")
