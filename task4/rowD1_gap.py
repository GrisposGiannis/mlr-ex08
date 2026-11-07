# Row D1 - global average pooling instead of flattening, on its own.
# Anchor: row 0a. The ONLY change is in the network: after the four convolution blocks the
# 64 x 6 x 6 map is averaged down to its 64 numbers (one number per channel) and fed straight
# to the output layer, instead of being flattened to 2304 numbers and passed through a linear
# layer with 64 neurons. That layer held 147 456 of the 208 292 parameters of LeafNet, so most
# of what the network could memorise with is gone, and what is left answers "is this feature
# present anywhere?", which is the question a leaf shape actually asks.
#
# A side effect worth noting: nothing in the network now depends on the input being exactly
# 96 x 96, because the averaging follows whatever size the map happens to have.
#
# Run:  SEED=42 python rowD1_gap.py

import torch
import torch.nn as nn

from common import SEED, accuracy, camera_val_loader, clean_transform, device, loader_for, log_row, plot_curves, \
    set_seed

EPOCHS = 15
set_seed(SEED)

# ------------------------------------------------------------ the data (Task 3a, unchanged)
train_ds, train_loader = loader_for("train", clean_transform, shuffle=True)
val_ds, val_loader = loader_for("val", clean_transform, shuffle=False)
hist_ds, hist_loader = loader_for("hist_val", clean_transform, shuffle=False)
camera_ds, camera_loader = camera_val_loader()
print(f"train {len(train_ds)} | val {len(val_ds)} | camera val {len(camera_ds)} | last year's val {len(hist_ds)}")
print(f"classes {train_ds.classes} | seed {SEED} | device {device}")


# ------------------------------------------------------------ the network: the one change
class LeafNetGAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)          # instead of x.flatten(1) and self.fc1
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.gap(x).flatten(1)                  # 64 numbers: the mean of every channel
        return self.fc2(x)


torch.manual_seed(SEED)
model = LeafNetGAP().to(device)
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
plot_curves(f"D1 global average pooling seed{SEED}", curve)
log_row("D1_global_average_pooling", SEED, EPOCHS, curve, mean_train_loss, best_epoch)
