import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ObjectDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = self.load_annotations()

    def load_annotations(self):
        annotations = []
        with open(os.path.join(self.root_dir, 'annotations.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                img_path = os.path.join(self.root_dir, 'images', parts[0])
                xmin, ymin, xmax, ymax = map(float, parts[1:])
                annotations.append((img_path, [xmin, ymin, xmax, ymax]))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, bbox = self.annotations[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(bbox, dtype=torch.float32)

# Example usage:
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ObjectDetectionDataset(root_dir='path_to_dataset', transform=data_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for images, bboxes in dataloader:
    # images: (batch_size, channels, height, width)
    # bboxes: (batch_size, 4) containing [xmin, ymin, xmax, ymax] for each bounding box
    print(images.shape, bboxes.shape)
