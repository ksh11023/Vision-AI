import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from custom_vit import VIT_Base_Deep  # Import your custom ViT model
import os

# Define data transforms for train and test sets
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet normalization
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet normalization
])

# Define data loaders
train_data = datasets.ImageFolder(root='train_data_path', transform=train_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.ImageFolder(root='test_data_path', transform=test_transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the model
model = VIT_Base_Deep()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        angle_output, product_output = model(inputs)
        loss_product = criterion(product_output, labels[:, 0])  # Product label loss
        loss_angle = criterion(angle_output, labels[:, 1])  # Camera angle label loss
        total_loss = loss_product + loss_angle  # Combine both losses
        total_loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item():.4f}')

# Evaluation
model.eval()
correct_product = 0
total_product = 0
correct_angle = 0
total_angle = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        angle_output, product_output = model(inputs)
        _, predicted_product = torch.max(product_output.data, 1)
        _, predicted_angle = torch.max(angle_output.data, 1)
        total_product += labels.size(0)
        total_angle += labels.size(0)
        correct_product += (predicted_product == labels[:, 0]).sum().item()
        correct_angle += (predicted_angle == labels[:, 1]).sum().item()

accuracy_product = 100 * correct_product / total_product
accuracy_angle = 100 * correct_angle / total_angle
print(f'Test Product Label Accuracy: {accuracy_product:.2f}%')
print(f'Test Camera Angle Label Accuracy: {accuracy_angle:.2f}%')
