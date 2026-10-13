from pathlib import Path
from PIL import Image

for source in Path(".").glob("*.pgm"):
    output = source.with_suffix(".png")
    Image.open(source).save(output)
    print(f"Saved {output}")
