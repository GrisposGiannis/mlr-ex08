# Row F2 - the same combination as F1, plus the colour augmentation.
# The only difference is the ColorJitter line in the training transform. Row A tested colour
# jitter on the small network from scratch, where it cost clean accuracy and its camera gain
# stayed inside the noise; this row asks the same question where it matters, on the pretrained
# network that will actually go into the phone, because colour is exactly what a real camera
# changes - cloud, lamp, shadow, an autumn leaf against a green one.
#
# Run:  SEED=42 python rowF2_combo_colour.py        EXPORT=1 also writes the app files
# on this machine

import os

import torch
import torch.nn as nn
from torchvision import transforms, models

from common import SEED, SIZE, accuracy, camera_val_loader, clean_transform, device, export_for_camera_app, \
    loader_for, log_row, plot_curves, set_seed

EPOCHS = 30
set_seed(SEED)

# ------------------------------------------------------------ the data: A0's augmentation + colour
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds, train_loader = loader_for("train", aug_transform, shuffle=True)
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

# ------------------------------------------------------------ the training of row C
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
    scheduler.step()

    curve["loss"].append(mean_train_loss)
    curve["val"].append(accuracy(model, val_loader))
    curve["camera"].append(accuracy(model, camera_loader))
    curve["hist"].append(accuracy(model, hist_loader))
    print(f"epoch {epoch + 1:2d} | train loss {mean_train_loss:.4f} | val {curve['val'][-1]:.3f} "
          f"| camera {curve['camera'][-1]:.3f} | last year {curve['hist'][-1]:.3f}")

best_epoch = max(range(EPOCHS), key=lambda e: curve["val"][e]) + 1
plot_curves(f"F2 combination with colour seed{SEED}", curve)
log_row("F2_combination_colour", SEED, EPOCHS, curve, mean_train_loss, best_epoch)

if os.environ.get("EXPORT"):
    export_for_camera_app(model, train_ds.classes, sample_image_path=val_ds.samples[0][0])
