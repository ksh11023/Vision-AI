import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class MultiLabelCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes_product = sorted(os.listdir(os.path.join(root_dir, 'product')))
        self.classes_angle = sorted(os.listdir(os.path.join(root_dir, 'angle')))
        self.class_to_idx_product = {cls_name: idx for idx, cls_name in enumerate(self.classes_product)}
        self.class_to_idx_angle = {cls_name: idx for idx, cls_name in enumerate(self.classes_angle)}
        self.samples = self.make_dataset()

    def make_dataset(self):
        samples = []
        product_dir = os.path.join(self.root_dir, 'product')
        angle_dir = os.path.join(self.root_dir, 'angle')
        for img_name in os.listdir(product_dir):
            img_path = os.path.join(product_dir, img_name)
            product_label = img_name.split('_')[1]  # Assuming the product label is part of the filename
            angle_label = img_name.split('_')[2]  # Assuming the angle label is part of the filename
            product_label_idx = self.class_to_idx_product[product_label]
            angle_label_idx = self.class_to_idx_angle[angle_label]
            samples.append((img_path, product_label_idx, angle_label_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, product_label_idx, angle_label_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([product_label_idx, angle_label_idx], dtype=torch.long)
