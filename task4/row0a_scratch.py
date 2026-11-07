# Row 0a - BASELINE of the from-scratch branch.
# The Task 3 pipeline exactly: LeafDataset + LeafNet trained from scratch, the clean
# transform, CrossEntropyLoss, Adam with lr 1e-3, 15 epochs. No augmentation, no batch
# normalization. This is notebook cells 54 + 56 + 58 with only the two extra validations
# (camera-like, last year's) added next to the clean one - measurement, not a change of
# the training.
# Anchor for candidate A (colour augmentation) and candidate D (regularisation, architecture).
#
# Run:  SEED=42 python row0a_scratch.py     (SEED=123, SEED=999 for the noise runs)

import torch
import torch.nn as nn

from common import SEED, accuracy, camera_val_loader, clean_transform, device, loader_for, log_row, plot_curves, \
    set_seed

EPOCHS = 15
set_seed(SEED)

# ------------------------------------------------------------ the data (Task 3a)
train_ds, train_loader = loader_for("train", clean_transform, shuffle=True)   # the baseline sees the clean images
val_ds, val_loader = loader_for("val", clean_transform, shuffle=False)
hist_ds, hist_loader = loader_for("hist_val", clean_transform, shuffle=False)
camera_ds, camera_loader = camera_val_loader()
print(f"train {len(train_ds)} | val {len(val_ds)} | camera val {len(camera_ds)} | last year's val {len(hist_ds)}")
print(f"classes {train_ds.classes} | seed {SEED} | device {device}")


# ------------------------------------------------------------ the network (Task 3b, unchanged)
class LeafNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


torch.manual_seed(SEED)
model = LeafNet().to(device)
print("parameters:", sum(p.numel() for p in model.parameters()))

# ------------------------------------------------------------ the training (Task 3c, unchanged)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

curve = {"loss": [], "val": [], "camera": [], "hist": []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    mean_train_loss = total_loss / len(train_ds)

    curve["loss"].append(mean_train_loss)
    curve["val"].append(accuracy(model, val_loader))
    curve["camera"].append(accuracy(model, camera_loader))
    curve["hist"].append(accuracy(model, hist_loader))
    print(f"epoch {epoch + 1:2d} | train loss {mean_train_loss:.4f} | val {curve['val'][-1]:.3f} "
          f"| camera {curve['camera'][-1]:.3f} | last year {curve['hist'][-1]:.3f}")

best_epoch = max(range(EPOCHS), key=lambda e: curve["val"][e]) + 1
plot_curves(f"0a scratch seed{SEED}", curve)
log_row("0a_scratch_baseline", SEED, EPOCHS, curve, mean_train_loss, best_epoch)
