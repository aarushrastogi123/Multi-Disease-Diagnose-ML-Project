# src/dataloader_setup.py
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.dataset import RetinalDataset
import pandas as pd

def get_dataloaders(train_csv="data/train.csv", train_img_dir="data/train",
                    val_csv="data/val_labels.csv", val_img_dir="data/val",
                    batch_size=32, val_split=0.2, num_workers=0):
    """
    Return (train_loader, val_loader).
    If val_csv exists and has labels, uses it; otherwise splits train.csv into train/val.
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # If val csv exists and non-empty, use separate val set
    use_val_csv = False
    if os.path.exists(val_csv):
        try:
            df_val = pd.read_csv(val_csv)
            if len(df_val) > 0 and "id_code" in df_val.columns:
                use_val_csv = True
        except Exception:
            use_val_csv = False

    if use_val_csv:
        train_dataset = RetinalDataset(train_csv, train_img_dir, transform=transform, has_labels=True)
        val_dataset = RetinalDataset(val_csv, val_img_dir, transform=transform, has_labels=True)
        print(f"Using provided val CSV. Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    else:
        # load full train and split
        full_dataset = RetinalDataset(train_csv, train_img_dir, transform=transform, has_labels=True)
        total = len(full_dataset)
        if total == 0:
            raise RuntimeError("No training images found. Check data/train and train.csv")
        val_size = int(total * val_split)
        train_size = total - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"Split training data. Total: {total} | Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
