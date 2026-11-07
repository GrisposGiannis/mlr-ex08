# Builds the folder the study trains on, out of the group's raw photographs:
#
#   task4/data/leaves/train/<class>/*.jpg     75 % of every class
#   task4/data/leaves/val/<class>/*.jpg       25 % of every class
#
# The raw folder is only ever READ, nothing in it is changed. For every photo:
#   1. it is turned the right way up first, while it is still full resolution (95 of the
#      131 photographs carry an EXIF rotation tag that Image.open().convert("RGB") ignores),
#   2. it is cut to its CENTRE SQUARE, which is exactly what index.html does to the camera
#      frame before it feeds the model, so the training images and the ones the phone shows
#      the network are made the same way,
#   3. it is shrunk to 192 x 192 - twice what the network eats, so that the random crop of
#      the augmentation still has something to look at, and a folder of photographs that is
#      small enough to open and inspect by hand,
#   4. the photographs of every class are shuffled with a FIXED seed and split 75 / 25.
#
# Near-duplicate photographs - the same leaf shot twice one second apart - are detected and
# only the first of each pair is kept: otherwise one leaf can end up in the training set and
# its twin in the validation set, and the validation accuracy then measures memory.
#
# Run:  /c/Users/Giannis.G/Desktop/MRL-ex08/_venv/Scripts/python.exe make_dataset.py

import os
import random

from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()      # so that Image.open can read the iPhone .HEIC files

RAW = r"C:\Users\Giannis.G\Downloads\leafs_data_group1-20261106T153640Z-1-001\leafs_data_group1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "leaves")

SEED = 42
STORE = 192          # the edge of the stored square photographs
TRAIN_FRACTION = 0.75
DUPLICATE_LIMIT = 20.0   # mean difference of the 16x16 grey signatures below which two
                         # photographs are taken to be the same leaf. Measured on this data
                         # set: a hand-held re-shot of the same leaf sits at 12.5
                         # (IMG_7817 next to IMG_7818), while two different leaves sit at a
                         # median of 32 - so 20 splits them and drops the twins, not the class


def prepared(path):
    """The photograph as the study sees it: right way up, centre square, STORE x STORE."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        wide, high = image.size
        edge = min(wide, high)
        image = image.crop(((wide - edge) // 2, (high - edge) // 2,
                            (wide - edge) // 2 + edge, (high - edge) // 2 + edge))
        return image.resize((STORE, STORE))


def signature(image):
    """A coarse grey thumbnail, as bytes: two photographs of the same leaf are close."""
    return list(image.convert("L").resize((16, 16)).tobytes())


def difference(one, other):
    return sum(abs(a - b) for a, b in zip(one, other)) / len(one)


rng = random.Random(SEED)

for cls in sorted(os.listdir(RAW)):
    cls_folder = os.path.join(RAW, cls)
    if not os.path.isdir(cls_folder):
        continue

    kept, kept_signatures, duplicates, unreadable = [], [], [], []
    for file_name in sorted(os.listdir(cls_folder)):
        path = os.path.join(cls_folder, file_name)
        if not os.path.isfile(path):
            continue
        try:
            image = prepared(path)
        except Exception as error:
            unreadable.append(f"{file_name} ({type(error).__name__})")
            continue
        # the duplicate test, written out plainly: is this photograph like one already kept?
        signature_of_image = signature(image)
        first_similar = -1
        for index, other in enumerate(kept_signatures):
            if difference(other, signature_of_image) < DUPLICATE_LIMIT:
                first_similar = index
                break
        if first_similar >= 0:
            duplicates.append(f"{file_name} ~= {kept[first_similar][0]}")
            continue
        kept.append((file_name, image))
        kept_signatures.append(signature_of_image)

    rng.shuffle(kept)
    n_train = round(len(kept) * TRAIN_FRACTION)
    for split, files in [("train", kept[:n_train]), ("val", kept[n_train:])]:
        split_folder = os.path.join(OUT, split, cls)
        os.makedirs(split_folder, exist_ok=True)
        for number, (file_name, image) in enumerate(files):
            # every photograph becomes a .jpg, so the loader never meets a .HEIC again
            stem = file_name.rsplit(".", 1)[0]
            image.save(os.path.join(split_folder, f"{number:02d}_{stem}.jpg"), quality=95)

    print(f"{cls:9s} raw {len(os.listdir(cls_folder)):3d} | kept {len(kept):3d} "
          f"| train {n_train:3d} | val {len(kept) - n_train:3d}"
          + (f" | duplicates dropped: {'; '.join(duplicates)}" if duplicates else "")
          + (f" | unreadable: {'; '.join(unreadable)}" if unreadable else ""))

print(f"\nwritten under {OUT}")
