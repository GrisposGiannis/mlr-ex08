# Row E - the combination of row F1, with last year's photographs added to the training set.
# The only difference is the training data: 89 photographs of our own group plus the 700 of the
# archive, other students, other phones, a different autumn. 89 training images is very little,
# so this asks the question that matters most for a phone classifier - does more data, from
# other people's cameras, beat any amount of cleverness about our own few photographs?
#
# The archive's own 120 validation photographs were never trained on, in this row or any other,
# so the "last year" column stays an honest outside test for every row of the study: here it is
# the only column that measures what the added data was supposed to bring.
#
# Run:  SEED=42 python rowE_plus_archive.py        EXPORT=1 also writes the app files

import os

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models

from common import BATCH, PATHS, SEED, LeafDataset, accuracy, camera_val_loader, clean_transform, device, \
    export_for_camera_app, geo_train_transform, loader_for, log_row, plot_curves, set_seed

EPOCHS = 30
set_seed(SEED)

# ------------------------------------------------------------ the data: the one change
group_train = LeafDataset(PATHS["train"], geo_train_transform)
archive_train = LeafDataset(PATHS["hist_train"], geo_train_transform)
train_ds = ConcatDataset([group_train, archive_train])          # was the group's 89 alone
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_ds, val_loader = loader_for("val", clean_transform, shuffle=False)
hist_ds, hist_loader = loader_for("hist_val", clean_transform, shuffle=False)
camera_ds, camera_loader = camera_val_loader()
print(f"train {len(train_ds)} ({len(group_train)} ours + {len(archive_train)} last year's) | "
      f"val {len(val_ds)} | camera val {len(camera_ds)} | last year's val {len(hist_ds)}")
print(f"classes {train_ds.datasets[0].classes} | seed {SEED} | device {device}")

# ------------------------------------------------------------ the network and the training, as row F1
model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
model = model.to(device)
print("parameters:", sum(p.numel() for p in model.parameters()))

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
plot_curves(f"E with last year's data seed{SEED}", curve)
log_row("E_plus_last_years_data", SEED, EPOCHS, curve, mean_train_loss, best_epoch)

if os.environ.get("EXPORT"):
    export_for_camera_app(model, group_train.classes, sample_image_path=val_ds.samples[0][0])
