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
# url = "https://raw.githubusercontent.com/google/dreambooth/main/dataset/dog6/02.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# image.save("/root/02.jpg")

images = []
for s in [1, 2, 4, 8, 16]:
    image = Image.open("/root/02.jpg").resize((1024, 1024))
    width, height = image.size
    image = image.resize((width//s, height//s))
    image = add_gaussian_noise(image, 0.9)
    images.append(image)

make_image_grid(images, 1, len(images), resize=512).save("/root/resolution.jpg")


from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import cv2

gamma = 0.9
width, height = 512, 512
noise1 = np.random.normal(0.0, 1.0, (height, width, 3))
noise2 = np.random.normal(0.0, 1.0, (height//2, width//2, 3))
noise2 = cv2.resize(noise2, (width, height), interpolation=cv2.INTER_NEAREST)
# noise2 = cv2.resize(noise2, (width, height), interpolation=cv2.INTER_LANCZOS4)
# noise2 = cv2.resize(noise2, (width, height), interpolation=cv2.INTER_CUBIC)

image = cv2.imread("/root/02.jpg")
image = cv2.resize(image, (512, 512))
image_array = image / 255.0

noisy_images = []
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    noise = np.sqrt(alpha) * noise1 + np.sqrt(1.0 - alpha) * noise2
    noisy_image_array: np.ndarray = np.clip(np.sqrt(gamma) * image_array + np.sqrt(1 - gamma) * noise, 0, 1.0)
    noisy_image_array = (255 * noisy_image_array).astype(np.uint8)
    noisy_images.append(noisy_image_array)
    # Fit a normal distribution to the data:
    mu, std = norm.fit(noise.flatten())
    print(f"mu={mu}, std^2={std**2}")
x = cv2.hconcat(noisy_images)
cv2.imwrite("/root/noised.jpg", cv2.hconcat(noisy_images))
exit()

# Generate some data for this demonstration.
# data = norm.rvs(10.0, 2.5, size=500)
data = np.sqrt(1 - gamma) * noise
data = data.flatten()
# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=50, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = r"Fit results: $\mu = %.2f$,  $\sigma^2 = %.2f$" % (mu, std**2)
plt.title(title)

plt.savefig("/root/norm.png")