import glob
from pathlib import Path

import fire
from PIL import Image


def main(num_epochs: int, seed: int):
    weight_dir = Path(f"tmp/weights/aae_swiss_roll/e{num_epochs}/s{seed}")
    frames = []
    images = sorted(
        weight_dir.glob("latent_e*.png"), key=lambda path: int(path.stem.rsplit("_e", 1)[1])
    )
    for image in images:
        with open(image, "rb") as file:
            img = Image.open(file)
            img.load()
            frames.append(img)

    output = weight_dir / f"latent_animation.gif"
    frames[0].save(
        str(output),
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )


if __name__ == "__main__":
    fire.Fire(main)
