# The part that is THE SAME for every Task 4 run: seeds, data, the three
# validation sets, the accuracy function, the curve plot and the results log.
# Nothing in here depends on the variant. Every row's own file holds only the
# model, the loss, the optimizer and the training loop, so that the difference
# between two rows is exactly the one change being tested.

import os
import random
import sys
import urllib.request
import zipfile

# The Windows console is cp1252 and torch's ONNX exporter prints characters it cannot encode,
# which would abort the export of step 2e. The same line as in analyze.py.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

# ---------------------------------------------------------------- the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- reproducibility
def set_seed(seed):
    """Every random source we use, fixed. Call once, before building anything."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- configuration
SEED = int(os.environ.get("SEED", 42))      # the noise runs use SEED=123, SEED=999
BATCH = 32
SIZE = 96

DATA_DIR = os.environ.get("LEAVES_DIR", "data")        # our group's photos:  DATA_DIR/leaves/{train,val}
HIST_DIR = os.environ.get("HIST_DIR", "data_last_year")  # last year's photos: HIST_DIR/leaves/{train,val}

# The archive of last year's photographs (175 + 30 per species, other students, other phones).
HIST_ZIP = ("https://raw.githubusercontent.com/tomasvicar/MLR-public/"
            "master/exercises/data/ex08_leaves_images.zip")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# The clean transform: Task 2a / Task 3a with one line added, Resize. The group's photographs
# arrive as full-resolution phone shots, and make_dataset.py stores them as 192 x 192 centre
# squares, so the network's 96 x 96 input has to be produced somewhere. Doing it here with a
# plain downscale is what index.html does at inference time (central square, resize to 96,
# normalize), so the model is measured on exactly what the phone will show it. On last year's
# archive, whose photographs are already 96 x 96, the Resize is a no-op.
clean_transform = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# The augmented training transform of Task 3d, unchanged: geometric only, no colour change.
geo_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ---------------------------------------------------------------- our Dataset (the class of Task 3a, unchanged)
class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform

        # The class names: the sorted subfolders of the folder.
        self.classes = sorted([
            entry for entry in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, entry))
        ])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # A list of (file path, class index) pairs: every class, its files in sorted order.
        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(folder, cls_name)
            for file_name in sorted(os.listdir(cls_folder)):
                file_path = os.path.join(cls_folder, file_name)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        file_path, class_idx = self.samples[i]
        image = Image.open(file_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, class_idx


# ---------------------------------------------------------------- the data sets
def ensure_last_year():
    """Download and unpack last year's archive the first time it is needed."""
    if not os.path.isdir(os.path.join(HIST_DIR, "leaves", "train")):
        os.makedirs(HIST_DIR, exist_ok=True)
        zip_path = os.path.join(HIST_DIR, "ex08_leaves_images.zip")
        urllib.request.urlretrieve(HIST_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(HIST_DIR)
    return os.path.join(HIST_DIR, "leaves")


last_year = ensure_last_year()

PATHS = {
    "train": os.path.join(DATA_DIR, "leaves", "train"),
    "val": os.path.join(DATA_DIR, "leaves", "val"),
    "hist_train": os.path.join(last_year, "train"),
    "hist_val": os.path.join(last_year, "val"),
}


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PLACEHOLDER, in force only while the group's own photographs are missing. As soon as
# DATA_DIR/leaves/train exists again - the teacher's ZIP unpacked, or LEAVES_DIR pointed at
# it - nothing below happens and every row is trained on the group's photographs as it should
# be. No line to edit back.
PLACEHOLDER = not os.path.isdir(PATHS["train"])
if PLACEHOLDER:
    PATHS["train"] = PATHS["hist_train"]
    PATHS["val"] = PATHS["hist_val"]
    print("!!! PLACEHOLDER: no group photographs, training and validating on LAST YEAR'S "
          f"archive instead ({PATHS['train']}).")
    print("!!! Every row of results.csv gets the prefix PLACEHOLDER_ and NONE of them belongs "
          "in the report. val and hist_val are the same images now, so those two columns match.")
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def loader_for(name, transform, shuffle):
    """One of train / val / hist_train / hist_val, as a (dataset, loader) pair."""
    dataset = LeafDataset(PATHS[name], transform)
    return dataset, DataLoader(dataset, batch_size=BATCH, shuffle=shuffle)


# ---------------------------------------------------------------- the camera-like validation set
# The clean validation photographs, seen through a phone camera: an off-centre crop,
# a tilt, a perspective slant and changed colours. Built ONCE with its own fixed seed and
# cached on the disk, so that every variant of the study is measured on exactly the same
# pixels and the difference between two rows is the change and not the noise of the draw.
CAMERA_SEED = 7
camera_aug = transforms.Compose([
    transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0)),
    transforms.RandomRotation(30),
    transforms.RandomPerspective(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def camera_val_loader():
    # The name carries how many images went into it, so a cache made from another
    # validation folder can never be taken for this one.
    n_images = len(LeafDataset(PATHS["val"], None))
    cache = f"camera_val_seed{CAMERA_SEED}_{n_images}images.pt"

    if os.path.exists(cache):
        images, labels = torch.load(cache)
        dataset = TensorDataset(images, labels)
        return dataset, DataLoader(dataset, batch_size=BATCH)

    # The random state of the training is remembered and put back afterwards: building the
    # camera set must not shift the random numbers the training itself will draw.
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    torch.manual_seed(CAMERA_SEED)
    np.random.seed(CAMERA_SEED)
    random.seed(CAMERA_SEED)

    # The photographs as they are on the disk: the augmentation above ends with ToTensor and
    # Normalize, so it receives the images untransformed and turns them into tensors itself.
    val_set = LeafDataset(PATHS["val"], None)
    images = torch.stack([camera_aug(Image.open(path).convert("RGB")) for path, _ in val_set.samples])
    labels = torch.tensor([index for _, index in val_set.samples])
    torch.save((images, labels), cache)

    torch.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)

    print(f"camera validation set built once from {len(labels)} images (seed {CAMERA_SEED}) -> {cache}")
    dataset = TensorDataset(images, labels)
    return dataset, DataLoader(dataset, batch_size=BATCH)


# ---------------------------------------------------------------- measurement
def accuracy(model, loader):
    """Validation accuracy of the model on one loader. Evaluation mode, no gradients."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


# ---------------------------------------------------------------- the log and the curves
RESULTS_CSV = "results.csv"
HEADER = "row,seed,epochs,val_acc,camera_val_acc,hist_val_acc,final_train_loss,best_epoch"


def log_row(name, seed, epochs, curve, final_train_loss, best_epoch):
    """Append the run to results.csv and print the same numbers as a Markdown table row."""
    if PLACEHOLDER:
        name = "PLACEHOLDER_" + name
    row = [name, str(seed), str(epochs), f"{curve['val'][-1]:.4f}", f"{curve['camera'][-1]:.4f}",
           f"{curve['hist'][-1]:.4f}", f"{final_train_loss:.4f}", str(best_epoch)]
    new_file = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a") as f:
        if new_file:
            f.write(HEADER + "\n")
        f.write(",".join(row) + "\n")
    print("\n| " + " | ".join(row) + " |")


# ---------------------------------------------------------------- the export of step 2e
def export_for_camera_app(model, classes, sample_image_path=None, name="leaf_model"):
    """The model in the form the course phone app loads, exactly as in Task 2e: the ONNX file
    with the preprocessing written into its metadata, the weights of this very run, and - when
    one validation photograph is handed to it - the self-test picture with the probabilities
    this model gives for it."""
    import json
    import onnx

    model.eval()
    model.cpu()
    torch.save(model.state_dict(), "best_model_weights.pt")
    torch.onnx.export(model, (torch.zeros(1, 3, SIZE, SIZE),), f"{name}.onnx",
                      input_names=["image"], output_names=["logits"], external_data=False)

    info = {"classes": classes, "input_size": SIZE, "mean": mean, "std": std}
    onnx_model = onnx.load(f"{name}.onnx")
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(info))
    onnx.save(onnx_model, f"{name}.onnx")
    with open("model_info.json", "w") as f:
        json.dump(info, f)

    if sample_image_path is not None:
        # 96 x 96, so the app feeds it to the model as it is and the self-test measures the
        # pipeline rather than a resize. The probabilities are computed from the saved file,
        # the way the page reads it, not from the image in memory.
        clean = Image.open(sample_image_path).convert("RGB")
        edge = min(clean.size)
        clean = clean.crop(((clean.size[0] - edge) // 2, (clean.size[1] - edge) // 2,
                            (clean.size[0] - edge) // 2 + edge, (clean.size[1] - edge) // 2 + edge))
        clean = clean.resize((SIZE, SIZE))
        clean.save("check_image.jpg", quality=95)
        x = clean_transform(Image.open("check_image.jpg").convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probabilities = model(x).softmax(dim=1)[0]
        with open("check_expected.json", "w") as f:
            json.dump(dict(zip(classes, probabilities.tolist())), f, indent=1)

    print(f"exported {name}.onnx ({os.path.getsize(name + '.onnx') / 1e6:.1f} MB), "
          f"best_model_weights.pt, model_info.json" + ("" if sample_image_path is None else ", check_image.jpg, check_expected.json"))
    return info


def plot_curves(name, curve):
    """The three curves of one run: the loss, the clean validation accuracy, the camera one."""
    epochs = range(1, len(curve["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, curve["loss"])
    axes[0].set_title(f"{name}: training loss")

    # The camera panel carries both curves: the clean accuracy and what the same model
    # does on the camera-like images and on last year's photographs.
    axes[1].plot(epochs, curve["val"])
    axes[1].set_title(f"{name}: clean validation accuracy")
    axes[2].plot(epochs, curve["camera"], label="camera-like")
    axes[2].plot(epochs, curve["hist"], "--", label="last year's photos")
    axes[2].legend()
    axes[2].set_title(f"{name}: camera accuracy")

    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
    for axis in axes[1:]:
        axis.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.show()
