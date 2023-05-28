import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
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

    if args.dataset == "caltech256":
        data = datasets.Caltech256("data", download=True, transform=preprocess)
    elif args.dataset == "oxfordiiitpet":
        data = datasets.OxfordIIITPet("data", download=True, transform=preprocess)
    print(f"The dataset contains {len(data)} images.")

    # Remove images with labels that are greater to args.n_labels
    if args.n_labels < 257:
        indices = [i for i, (_, label) in enumerate(data) if label < args.n_labels]
        data = Subset(data, indices)

    if args.model == "resnet":
        model = torch.hub.load(
            "pytorch/vision", "resnet34", weights="ResNet34_Weights.IMAGENET1K_V1"
        )
        is_vit = False
    elif args.model == "vit":
        model = torch.hub.load(
            "pytorch/vision", "vit_b_16", weights="ViT_B_16_Weights.IMAGENET1K_V1"
        )
        is_vit = True

    filename = f"{args.model}_{args.dataset}_{args.file_extend}"

    model = replace_last_layers(
        model, args.layers, is_vit, unfreeze_norm=args.unfreeze_norm
    )

    if args.model == "resnet":
        if args.unfreeze_blocks < 0 or args.unfreeze_blocks > 4:
            raise ValueError("Unfreeze blocks must be between 0 and 4")
        if args.unfreeze_blocks > 0:
            for param in model.layer4.parameters():
                param.requires_grad = True

        if args.unfreeze_blocks > 1:
            for param in model.layer3.parameters():
                param.requires_grad = True

        if args.unfreeze_blocks > 2:
            for param in model.layer2.parameters():
                param.requires_grad = True

        if args.unfreeze_blocks > 3:
            for param in model.layer1.parameters():
                param.requires_grad = True

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_acc, val_acc = train(
        model, loss_fn, optimizer, train_loader, val_loader, epochs=args.epochs
    )
    test_acc = test_model(model, test_loader)

    plot_accurcies(train_acc, val_acc, filename=f"{filename}.png")
    store_final_accuracies(train_acc, val_acc, test_acc, filename=f"{filename}_acc.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="caltech256")
    parser.add_argument(
        "--model", type=str, default="resnet", choices=["resnet", "vit"]
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layers", default=[], action="append", type=int)
    parser.add_argument("--file_extend", type=str, default="")
    parser.add_argument("--unfreeze_norm", type=bool, default=True)
    parser.add_argument("--unfreeze_blocks", type=int, default=0)
    parser.add_argument("--n_labels", type=int, default=257)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    args.layers.append(args.n_labels)
    main(args)
