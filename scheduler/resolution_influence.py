from typing import List
import requests
from PIL import Image
import numpy as np

def make_image_grid(images: List[Image.Image], rows: int, cols: int, resize: int = None) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



def add_gaussian_noise(image: Image.Image, gamma: float = 0.7) -> Image.Image:
    image_array = np.array(image) / 255.0
    noise = np.random.normal(0.0, 1.0, image_array.shape)
    noisy_image_array: np.ndarray = np.clip(np.sqrt(gamma) * image_array + np.sqrt(1 - gamma) * noise, 0, 1.0)
    noisy_image_array = (255 * noisy_image_array).astype(np.uint8)
    return Image.fromarray(noisy_image_array)

# url = "http://images.cocodataset.org/train2017/000000000034.jpg"
url = "https://raw.githubusercontent.com/google/dreambooth/main/dataset/dog6/02.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image.save("/root/02.jpg")

images = []
for s in [1, 2, 4, 8, 16]:
    image = Image.open("/root/02.jpg").resize((1024, 1024))
    width, height = image.size
    image = image.resize((width//s, height//s))
    image = add_gaussian_noise(image, 0.9)
    images.append(image)

make_image_grid(images, 1, len(images), resize=256).save("/root/resolution.jpg")