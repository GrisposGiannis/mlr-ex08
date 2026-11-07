# Row C - the pretrained network with a cosine learning-rate schedule and more epochs.
# Anchor: row 0b. The changes: EPOCHS 15 -> 30, and CosineAnnealingLR, which slides the
# learning rate from 1e-3 down to 0 along a half cosine over the thirty epochs. The large
# steps at the start do the moving, the small ones at the end let the weights settle into a
# minimum instead of bouncing across it - which is what the row 0b curve looks like when the
# validation accuracy wanders down over its last four epochs.
#
# Run:  SEED=42 python rowC_cosine.py

import torch
import torch.nn as nn
from torchvision import models

from common import SEED, accuracy, camera_val_loader, clean_transform, device, loader_for, log_row, plot_curves, \
    set_seed

EPOCHS = 30                      # was 15, the second half of this row's change
set_seed(SEED)

# ------------------------------------------------------------ the data (as row 0b)
train_ds, train_loader = loader_for("train", clean_transform, shuffle=True)
val_ds, val_loader = loader_for("val", clean_transform, shuffle=False)
hist_ds, hist_loader = loader_for("hist_val", clean_transform, shuffle=False)
camera_ds, camera_loader = camera_val_loader()
print(f"train {len(train_ds)} | val {len(val_ds)} | camera val {len(camera_ds)} | last year's val {len(hist_ds)}")
print(f"classes {train_ds.classes} | seed {SEED} | device {device}")

# ------------------------------------------------------------ the network (Task 2b, unchanged)
model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
model = model.to(device)
print("parameters:", sum(p.numel() for p in model.parameters()))

# ------------------------------------------------------------ the training + the scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
    scheduler.step()             # the one line inside the loop: one step of the schedule per epoch

    curve["loss"].append(mean_train_loss)
    curve["val"].append(accuracy(model, val_loader))
    curve["camera"].append(accuracy(model, camera_loader))
    curve["hist"].append(accuracy(model, hist_loader))
    print(f"epoch {epoch + 1:2d} | lr {scheduler.get_last_lr()[0]:.5f} | train loss {mean_train_loss:.4f} "
          f"| val {curve['val'][-1]:.3f} | camera {curve['camera'][-1]:.3f} | last year {curve['hist'][-1]:.3f}")

best_epoch = max(range(EPOCHS), key=lambda e: curve["val"][e]) + 1
plot_curves(f"C cosine seed{SEED}", curve)
log_row("C_cosine_30_epochs", SEED, EPOCHS, curve, mean_train_loss, best_epoch)
