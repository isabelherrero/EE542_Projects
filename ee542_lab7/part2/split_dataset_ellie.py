
from pathlib import Path
from PIL import Image
import random
import shutil

SOURCE = Path("data/original")
DEST = Path("data")
TRAIN_RATIO = 0.8

rng = random.Random(42)

for category in ["Cat", "Dog"]:
    src = SOURCE / category

    valid_images = []

    for path in src.iterdir():
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            with Image.open(path) as img:
                img.verify()
            valid_images.append(path)
        except Exception:
            print(f"Skipping corrupted image: {path.name}")

    rng.shuffle(valid_images)

    split = int(len(valid_images) * TRAIN_RATIO)

    train_images = valid_images[:split]
    test_images = valid_images[split:]

    for subset, images in [
        ("train", train_images),
        ("test", test_images)
    ]:
        output_dir = DEST / subset / category.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        for image in images:
            shutil.copy2(image, output_dir / image.name)

        print(f"{subset}/{category.lower()}: {len(images)} images")

print("Dataset split completed!")
