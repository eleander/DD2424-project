import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets
import argparse

from utils import (
    plot_accurcies,
    test_model,
    train,
    replace_last_layers,
    preprocess,
    store_final_accuracies,
)


def main(args):
    print(args)

    data = datasets.Caltech256("data", download=True, transform=preprocess)
    print(f"The dataset contains {len(data)} images.")

    if args.model == "resnet":
        model = torch.hub.load(
            "pytorch/vision", "resnet34", weights="ResNet34_Weights.IMAGENET1K_V1"
        )
        is_vit = False
        filename = "resnet34_caltech256"
    elif args.model == "vit":
        model = torch.hub.load(
            "pytorch/vision", "vit_b_16", weights="ViT_B_16_Weights.IMAGENET1K_V1"
        )
        is_vit = True
        filename = "vit_b_16_caltech256"

    filename += args.file_extend

    model = replace_last_layers(
        model, args.layers, is_vit, unfreeze_norm=args.unfreeze_norm
    )

    # Split the dataset into train, validation and test sets
    train_data, val_data, test_data = torch.utils.data.random_split(
        data, [0.6, 0.2, 0.2]
    )

    # Create data loaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_acc, val_acc = train(
        model, loss_fn, optimizer, train_loader, val_loader, epochs=args.epochs
    )
    test_acc = test_model(model, test_loader)

    plot_accurcies(train_acc, val_acc, filename=f"{filename}.png")
    store_final_accuracies(
        train_acc[-1], val_acc[-1], test_acc, filename=f"{filename}_acc.txt"
    )
    print(
        f"Accuracies:\n - Train: {train_acc[-1]:.3f}\n - Validation: {val_acc[-1]:.3f}\n - Test: {test_acc:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet", choices=["resnet", "vit"]
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layers", action="append", type=int)
    parser.add_argument("--file_extend", type=str, default="")
    parser.add_argument("--unfreeze_norm", type=bool, default=True)
    args = parser.parse_args()
    args.layers.append(257)
    main(args)
