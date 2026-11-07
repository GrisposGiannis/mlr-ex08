# Row B - the pretrained network with the backbone FROZEN: only the new last layer trains.
# Anchor: row 0b, where the whole network is fine-tuned. The ONLY change is the three lines
# under "the one change" and the matching optimizer:
#   every weight of the pretrained MobileNetV3-Small stops taking gradients, the four weights
#   of the new output layer keep theirs, and the frozen part is put in evaluation mode.
#
# The evaluation mode is not decoration. MobileNetV3 normalises inside its blocks, and a layer
# in training mode recomputes those statistics from the current batch and updates its stored
# ones - so a "frozen" backbone that is left in training mode is still rewritten by our 89
# photographs. In evaluation mode it keeps the statistics it learned from a million images,
# which is the whole point of freezing it: it becomes a fixed feature extractor.
#
# Run:  SEED=42 python rowB_frozen.py

import torch
import torch.nn as nn
from torchvision import models

from common import SEED, accuracy, camera_val_loader, clean_transform, device, loader_for, log_row, plot_curves, \
    set_seed

EPOCHS = 15
set_seed(SEED)

# ------------------------------------------------------------ the data (as row 0b)
train_ds, train_loader = loader_for("train", clean_transform, shuffle=True)
val_ds, val_loader = loader_for("val", clean_transform, shuffle=False)
hist_ds, hist_loader = loader_for("hist_val", clean_transform, shuffle=False)
camera_ds, camera_loader = camera_val_loader()
print(f"train {len(train_ds)} | val {len(val_ds)} | camera val {len(camera_ds)} | last year's val {len(hist_ds)}")
print(f"classes {train_ds.classes} | seed {SEED} | device {device}")

# ------------------------------------------------------------ the network (Task 2b) + the one change
model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
model = model.to(device)

for parameter in model.parameters():
    parameter.requires_grad = False           # nothing learns ...
for parameter in model.classifier[3].parameters():
    parameter.requires_grad = True            # ... except the new output layer
trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
print(f"parameters {sum(p.numel() for p in model.parameters())}, of which trainable "
      f"{sum(p.numel() for p in trainable)}")

# ------------------------------------------------------------ the training (Task 2c, over the trainable weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(trainable, lr=1e-3)

curve = {"loss": [], "val": [], "camera": [], "hist": []}

for epoch in range(EPOCHS):
    model.train()
    model.features.eval()                     # the frozen part keeps its pretrained statistics

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
plot_curves(f"B frozen backbone seed{SEED}", curve)
log_row("B_frozen_backbone", SEED, EPOCHS, curve, mean_train_loss, best_epoch)
