import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

EPOCHS = 50


class ResNetTransfer(nn.Module):
    def __init__(self, unfreeze_bn=False):
        super(ResNetTransfer, self).__init__()
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze BatchNorm layers if specified
        if unfreeze_bn:
            for module in self.resnet.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = True
                    for param in module.parameters():
                        param.requires_grad = True

        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.resnet(x)


def train_model(rank, world_size, unfreeze_bn=False):
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)

    # Add timeout and robust initialization
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        print(f"Process group initialization failed: {e}")
        raise e

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    data_path = os.path.expanduser("~/data/imagenette2_160")
    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, "imagenette2-160/train"), transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_path, "imagenette2-160/val"), transform=test_transform
    )

    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=128, num_workers=8, pin_memory=True
    )

    # Initialize model
    model = ResNetTransfer(unfreeze_bn=unfreeze_bn).cuda(rank)

    try:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
        )
    except Exception as e:
        print(f"DDP initialization failed: {e}")
        dist.destroy_process_group()
        raise e

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Different learning rates for different parts of the network
    fc_params = list(model.module.resnet.fc.parameters())
    bn_params = []
    if unfreeze_bn:
        bn_params = [
            param
            for name, param in model.named_parameters()
            if "bn" in name and param.requires_grad
        ]

    optimizer = optim.AdamW(
        [{"params": fc_params, "lr": 0.001}, {"params": bn_params, "lr": 0.0001}],
        weight_decay=0.01,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[0.001, 0.0001],
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # Training metrics
    metrics = {
        "train_losses": [],
        "test_losses": [],
        "train_accs": [],
        "test_accs": [],
        "confusion_matrix": None,
    }

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        if rank == 0:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        else:
            train_pbar = train_loader

        train_sampler.set_epoch(epoch)

        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if rank == 0:
                train_pbar.set_postfix(
                    {
                        "loss": f"{train_loss/total:.3f}",
                        "acc": f"{100.*correct/total:.2f}%",
                    }
                )

        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        if rank == 0:
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Test]")
        else:
            test_pbar = test_loader

        with torch.no_grad():
            for inputs, targets in test_pbar:
                inputs, targets = inputs.cuda(rank), targets.cuda(rank)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                if rank == 0:
                    test_pbar.set_postfix(
                        {
                            "loss": f"{test_loss/total:.3f}",
                            "acc": f"{100.*correct/total:.2f}%",
                        }
                    )

        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * correct / total

        if rank == 0:
            metrics["train_losses"].append(train_loss)
            metrics["test_losses"].append(test_loss)
            metrics["train_accs"].append(train_acc)
            metrics["test_accs"].append(test_acc)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}")
            print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}\n")

    if rank == 0:
        metrics["confusion_matrix"] = confusion_matrix(all_targets, all_preds)
        torch.save(metrics, f'resnet_results_bn{"_unfrozen" if unfreeze_bn else ""}.pt')

    dist.destroy_process_group()
    return metrics if rank == 0 else None


def plot_comparison():
    # Load results
    resnet_frozen = torch.load("resnet_results_bn.pt")
    resnet_unfrozen = torch.load("resnet_results_bn_unfrozen.pt")
    custom_model = torch.load("results_all_methods.pt")  # from Task 2

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy comparison
    ax1.plot(resnet_frozen["train_accs"], label="ResNet (Frozen BN) - Train")
    ax1.plot(resnet_frozen["test_accs"], label="ResNet (Frozen BN) - Test")
    ax1.plot(resnet_unfrozen["train_accs"], label="ResNet (Unfrozen BN) - Train")
    ax1.plot(resnet_unfrozen["test_accs"], label="ResNet (Unfrozen BN) - Test")
    ax1.plot(custom_model["train_accs"], label="Custom Model - Train")
    ax1.plot(custom_model["test_accs"], label="Custom Model - Test")

    ax1.set_title("Accuracy Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True)

    # Loss comparison
    ax2.plot(resnet_frozen["train_losses"], label="ResNet (Frozen BN) - Train")
    ax2.plot(resnet_frozen["test_losses"], label="ResNet (Frozen BN) - Test")
    ax2.plot(resnet_unfrozen["train_losses"], label="ResNet (Unfrozen BN) - Train")
    ax2.plot(resnet_unfrozen["test_losses"], label="ResNet (Unfrozen BN) - Test")
    ax2.plot(custom_model["train_losses"], label="Custom Model - Train")
    ax2.plot(custom_model["test_losses"], label="Custom Model - Test")

    ax2.set_title("Loss Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("model_comparison.png")

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    models = {
        "ResNet (Frozen BN)": resnet_frozen["confusion_matrix"],
        "ResNet (Unfrozen BN)": resnet_unfrozen["confusion_matrix"],
        "Custom Model": custom_model["confusion_matrix"],
    }

    for ax, (name, cm) in zip(axes, models.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig("confusion_matrices_comparison.png")


def main():
    world_size = torch.cuda.device_count()

    try:
        # Train with frozen BatchNorm
        print("\nTraining ResNet-18 with frozen BatchNorm layers...")
        mp.spawn(train_model, args=(world_size, False), nprocs=world_size, join=True)

        # Train with unfrozen BatchNorm
        print("\nTraining ResNet-18 with unfrozen BatchNorm layers...")
        mp.spawn(train_model, args=(world_size, True), nprocs=world_size, join=True)

        # Plot comparisons
        plot_comparison()

    except Exception as e:
        print(f"Training failed: {e}")
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e


if __name__ == "__main__":
    main()
