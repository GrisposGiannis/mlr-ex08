# Row D - the regularisation and architecture package, as the exercise lists it:
#   global average pooling instead of flattening  (= row D1)
#   dropout 0.5 after the hidden layer            (= row D2, here after the pooled features)
#   AdamW with weight decay 1e-4 instead of Adam
#   label smoothing 0.1 in the loss
#   keeping the weights of the best epoch instead of the last one
#
# Four changes at once, so on its own this row says only "the package helps or it does not".
# Rows D1 and D2 take it apart, and the report says which of the parts did the work.
#
# The best epoch is chosen on the CLEAN validation accuracy - the only one of the three sets a
# group has without going outside with a phone - so the camera column stays an untouched
# measurement rather than something we tuned on. The curve is cut at that epoch: what is
# reported is the model you would have shipped, which is the weights of that epoch.
#
# Run:  SEED=42 python rowD_full.py

import copy

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


# ------------------------------------------------------------ the network: pooling and dropout
class LeafNetRegularised(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)            # instead of x.flatten(1) and self.fc1
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc2(x)


torch.manual_seed(SEED)
model = LeafNetRegularised().to(device)
print("parameters:", sum(p.numel() for p in model.parameters()))

# ------------------------------------------------------------ the training: the other three changes
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)          # was nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)   # was Adam

curve = {"loss": [], "val": [], "camera": [], "hist": []}
best_state, best_val, best_epoch = None, 0.0, 0

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

    if curve["val"][-1] > best_val:                       # keep the weights of the best epoch
        best_val, best_epoch = curve["val"][-1], epoch + 1
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
curve = {key: values[:best_epoch] for key, values in curve.items()}
mean_train_loss = curve["loss"][-1]

plot_curves(f"D regularisation package seed{SEED}", curve)
log_row("D_regularisation_package", SEED, len(curve["loss"]), curve, mean_train_loss, best_epoch)
