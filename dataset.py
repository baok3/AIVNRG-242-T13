import os
import glob
import torch
from torch.utils.data import Dataset
import cv2


class RareDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.image_paths = []
        self.labels = []

        # Duyệt qua các thư mục con
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)

                # Lấy tất cả các file .jpg
                image_files = glob.glob(os.path.join(class_path, "*.jpg"))
                self.image_paths.extend(image_files)
                self.labels.extend([class_idx] * len(image_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
