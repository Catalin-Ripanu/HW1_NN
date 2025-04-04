import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import tarfile
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# Constants - utilizing GPU memory while respecting original image size
BATCH_SIZE = 128
NUM_WORKERS = 8
IMAGE_SIZE = 160
EPOCHS = 50


class GradientTracker:
    def __init__(self, rank):
        self.rank = rank
        self.current_epoch_grads = defaultdict(list)
        self.grad_stats = defaultdict(list)

    def update(self, model, epoch):
        """
        Collect gradient statistics for the current batch.
        Only rank 0 process will store the statistics.
        """
        if self.rank == 0:
            for name, param in model.module.named_parameters():
                if "weight" in name and param.grad is not None:
                    # Store statistics instead of raw gradients
                    grad_np = param.grad.data.cpu().numpy()
                    self.current_epoch_grads[name].append(
                        {
                            "mean": np.mean(grad_np),
                            "std": np.std(grad_np),
                            "max": np.max(np.abs(grad_np)),
                            "min": np.min(np.abs(grad_np)),
                            "median": np.median(np.abs(grad_np)),
                        }
                    )

    def epoch_end(self, epoch):
        """
        Process the accumulated gradient statistics at the end of each epoch.
        """
        if self.rank == 0:
            for name in self.current_epoch_grads:
                epoch_stats = {
                    "mean": np.mean(
                        [batch["mean"] for batch in self.current_epoch_grads[name]]
                    ),
                    "std": np.mean(
                        [batch["std"] for batch in self.current_epoch_grads[name]]
                    ),
                    "max": np.max(
                        [batch["max"] for batch in self.current_epoch_grads[name]]
                    ),
                    "min": np.min(
                        [batch["min"] for batch in self.current_epoch_grads[name]]
                    ),
                    "median": np.mean(
                        [batch["median"] for batch in self.current_epoch_grads[name]]
                    ),
                }
                self.grad_stats[name].append(epoch_stats)

            # Clear current epoch gradients
            self.current_epoch_grads.clear()

    def save_visualization(self, output_dir="gradient_plots"):
        """
        Create visualization of gradient statistics across epochs.
        Only called by rank 0 process at the end of training.
        """
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

            for name in self.grad_stats:
                epochs = range(1, len(self.grad_stats[name]) + 1)

                plt.figure(figsize=(15, 10))

                # Plot mean and std
                plt.subplot(2, 1, 1)
                means = [stats["mean"] for stats in self.grad_stats[name]]
                stds = [stats["std"] for stats in self.grad_stats[name]]
                plt.plot(epochs, means, label="Mean", marker="o")
                plt.fill_between(
                    epochs,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.3,
                )
                plt.title(f"Gradient Statistics Over Epochs - {name}")
                plt.xlabel("Epoch")
                plt.ylabel("Gradient Value")
                plt.legend()
                plt.grid(True)

                # Plot min/max/median
                plt.subplot(2, 1, 2)
                plt.plot(
                    epochs,
                    [stats["max"] for stats in self.grad_stats[name]],
                    label="Max",
                    marker="v",
                )
                plt.plot(
                    epochs,
                    [stats["median"] for stats in self.grad_stats[name]],
                    label="Median",
                    marker="s",
                )
                plt.plot(
                    epochs,
                    [stats["min"] for stats in self.grad_stats[name]],
                    label="Min",
                    marker="^",
                )
                plt.xlabel("Epoch")
                plt.ylabel("Gradient Magnitude")
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"gradients_{name}.png"))
                plt.close()


class ImagenetteNet(nn.Module):
    def __init__(self, use_bn=False, use_dropout=False):
        super(ImagenetteNet, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128) if use_bn else nn.Identity()

        # Conv Layer 2
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256) if use_bn else nn.Identity()

        # Conv Layer 3
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512) if use_bn else nn.Identity()

        # Calculate feature size after pooling
        feature_size = 512 * 20 * 20

        # Linear Layer 1
        self.fc1 = nn.Linear(feature_size, 1024)
        self.dropout1 = nn.Dropout(0.5) if use_dropout else nn.Identity()

        # Linear Layer 2 (Output)
        self.fc2 = nn.Linear(1024, 10)

        # Pooling layer (not counted in layer depth)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = torch.flatten(x, 1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


def get_data_transforms(use_augmentation=False):
    """
    Modified transforms to ensure consistent image sizes
    """
    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Force square size
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Force square size
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Force square size
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return train_transform, test_transform


def extract_dataset(data_path):
    """Extract the dataset if not already extracted"""
    if not os.path.exists(os.path.join(data_path, "imagenette2-160/train")):
        with tarfile.open(
            os.path.join(data_path, "imagenette2-160.tgz"), "r:gz"
        ) as tar:
            tar.extractall(path=data_path)


def train_model(rank, world_size, model_config):
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Extract dataset if needed
    data_path = os.path.expanduser("~/data/imagenette2_160/")
    extract_dataset(data_path)

    # Data loading
    train_transform, test_transform = get_data_transforms(
        model_config["use_augmentation"]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, "imagenette2-160/train"), transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_path, "imagenette2-160/val"), transform=test_transform
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

    # Model setup
    model = ImagenetteNet(
        use_bn=model_config["use_bn"], use_dropout=model_config["use_dropout"]
    ).cuda(rank)

    model = DDP(model, device_ids=[rank])

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training metrics
    metrics = {
        "train_losses": [],
        "test_losses": [],
        "train_accs": [],
        "test_accs": [],
        "confusion_matrix": None,
    }

    gradient_tracker = GradientTracker(rank)

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

        for inputs, targets in train_pbar:
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            gradient_tracker.update(model, epoch)

            optimizer.step()

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

        gradient_tracker.epoch_end(epoch)

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
        gradient_tracker.save_visualization()

        metrics["confusion_matrix"] = confusion_matrix(all_targets, all_preds)
        torch.save(metrics, f'results_{model_config["name"]}.pt')

    dist.destroy_process_group()
    return metrics if rank == 0 else None


def plot_comparison(results, comparison_key, title):
    plt.figure(figsize=(12, 5))

    # Training/Test Accuracy subplot
    plt.subplot(1, 2, 1)
    for name in results:
        if comparison_key in name or name == "baseline":
            plt.plot(results[name]["train_accs"], label=f"{name} (Train)")
            plt.plot(results[name]["test_accs"], label=f"{name} (Test)")
    plt.title(f"Accuracy Comparison - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Training/Test Loss subplot
    plt.subplot(1, 2, 2)
    for name in results:
        if comparison_key in name or name == "baseline":
            plt.plot(results[name]["train_losses"], label=f"{name} (Train)")
            plt.plot(results[name]["test_losses"], label=f"{name} (Test)")
    plt.title(f"Loss Comparison - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'comparison_{title.lower().replace(" ", "_")}.png')
    plt.close()


def main():
    world_size = torch.cuda.device_count()

    try:
        configs = [
            {
                "name": "baseline",
                "use_bn": False,
                "use_dropout": False,
                "use_augmentation": False,
            },
            {
                "name": "with_bn",
                "use_bn": True,
                "use_dropout": False,
                "use_augmentation": False,
            },
            {
                "name": "with_dropout",
                "use_bn": False,
                "use_dropout": True,
                "use_augmentation": False,
            },
            {
                "name": "with_augmentation",
                "use_bn": False,
                "use_dropout": False,
                "use_augmentation": True,
            },
            {
                "name": "all_methods",
                "use_bn": True,
                "use_dropout": True,
                "use_augmentation": True,
            },
        ]

        results = {}
        for config in configs:
            print(f"\nTraining model with configuration: {config['name']}")
            mp.spawn(
                train_model, args=(world_size, config), nprocs=world_size, join=True
            )
            results[config["name"]] = torch.load(f'results_{config["name"]}.pt')

        # Plot comparisons
        plot_comparison(results, "bn", "Batch Normalization")
        plot_comparison(results, "dropout", "Dropout")
        plot_comparison(results, "augmentation", "Data Augmentation")
        plot_comparison(
            {"baseline": results["baseline"], "all_methods": results["all_methods"]},
            "all",
            "All Methods",
        )

        # Plot confusion matrices
        class_names = [
            "tench",
            "springer",
            "cassette",
            "chainsaw",
            "church",
            "french_horn",
            "garbage_truck",
            "gas_pump",
            "golf_ball",
            "parachute",
        ]

        for name, data in results.items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                data["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"confusion_matrix_{name}.png")
            plt.close()

    except Exception as e:
        print(f"Training failed: {e}")
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e


if __name__ == "__main__":
    main()
