from PIL import Image

img = Image.open("input.png").convert("L")

for m in [512, 1024, 2048]:
    out = img.resize((m, m))

    name = f"image_{m}.tiff"

    out.save(
        name,
        format="TIFF",
        compression="raw"
    )

    print("Created", name)