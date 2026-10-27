import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FashionMNIST(
    root="data", #which folder to download to
    train=True,  # get training data
    download=True, # download data if not yet
    transform=ToTensor(), # images come as PIL format
    target_transform=None # you can transform labels
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
import matplotlib.pyplot as plt
image, label = train_data[0]
print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze(), cmap="gray")
plt.title(train_data.classes[label])
plt.axis(False)
plt.show()

from torch.utils.data import DataLoader
BATCH_SIZE : int = 32
train_loader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_loader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
print(f"Dataloaders: {train_loader, test_loader}") 
print(f"Length of train dataloader: {len(train_loader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_loader)} batches of {BATCH_SIZE}")

from torch import nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train
for epoch in range(5):
    model.train()
    for X, y in train_loader:
        loss = loss_fn(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")


model.eval()
correct = 0
with torch.no_grad():                     # no gradients needed, faster and less memory
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)   # index of highest score = predicted digit
        correct += (preds == labels).sum().item()
print(correct / len(test_data))