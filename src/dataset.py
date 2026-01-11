# src/dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RetinalDataset(Dataset):
    """
    Dataset that reads a csv with columns ['id_code','diagnosis'] (for train/val)
    or a csv with only ['id_code'] for test (no labels).
    img_dir should be the folder containing images (files named id_code + .png/.jpg).
    """
    def __init__(self, csv_file, img_dir, transform=None, has_labels=True, img_exts=(".png", ".jpg", ".jpeg")):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        self.img_exts = img_exts

        if csv_file is None:
            raise ValueError("csv_file must be provided (path to train.csv/test.csv)")

        self.df = pd.read_csv(csv_file)
        # If test.csv has only id_code, has_labels=False
        if has_labels and "diagnosis" not in self.df.columns:
            raise ValueError("CSV expected to have 'diagnosis' column when has_labels=True")

        # build list of file paths and labels
        self.image_paths = []
        self.labels = []

        for idx, row in self.df.iterrows():
            id_code = str(row['id_code'])
            # try extensions in order
            found = False
            for ext in self.img_exts:
                path = os.path.join(self.img_dir, id_code + ext)
                if os.path.exists(path):
                    self.image_paths.append(path)
                    found = True
                    break
            if not found:
                # not fatal — skip but log later
                continue
            if has_labels:
                self.labels.append(int(row['diagnosis']))

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {img_dir} matching ids in {csv_file}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.has_labels:
            label = self.labels[idx]
            return image, label
        else:
            return image, os.path.basename(path)  # return filename for test
