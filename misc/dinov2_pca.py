"""
wget https://github.com/MartinBurian/dinov2/blob/experiments/experiments/data/crane/crane1.jpg
wget https://github.com/MartinBurian/dinov2/blob/experiments/experiments/data/crane/crane2.jpg
wget https://github.com/MartinBurian/dinov2/blob/experiments/experiments/data/crane/crane3.jpg
wget https://github.com/MartinBurian/dinov2/blob/experiments/experiments/data/crane/crane4.jpg
"""

from PIL import Image
import numpy as np
import cv2

import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from transformers import Dinov2Model


transform = T.Compose([
    T.Resize(448, interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

model = Dinov2Model.from_pretrained("facebook/dinov2-base")

pixel_values = torch.stack([transform(Image.open(f"/root/crane{i+1}.jpg")) for i in range(4)], dim=0)
print(pixel_values.shape)
pixel_values = pixel_values.cuda()
model.cuda()
with torch.no_grad():
    outputs = model(pixel_values=pixel_values, output_hidden_states=True)
    # patch_tokens: torch.Tensor = outputs.hidden_states[-1][:, 1:, :]
    patch_tokens: torch.Tensor = outputs.last_hidden_state[:, 1:, :]

patch_tokens = patch_tokens.cpu().numpy()
patch_w = patch_h = int(patch_tokens.shape[1]**0.5)
features = patch_tokens.reshape(4 * patch_h * patch_w, -1)

pca = PCA(n_components=3)
scaler = MinMaxScaler(clip=True)
# PCA Feature
pca_features = pca.fit_transform(features)
pca_features = scaler.fit_transform(pca_features)

is_foreground_larger_than_threshold = False
background_threshold = 0.3

# Foreground/Background
if is_foreground_larger_than_threshold:
    pca_features_bg = pca_features[:, 0] < background_threshold
else:
    pca_features_bg = pca_features[:, 0] > background_threshold
pca_features_fg = ~ pca_features_bg

# PCA with only foreground
pca_features_rem = pca.fit_transform(features[pca_features_fg])

# Min Max Normalization
pca_features_rem = scaler.fit_transform(pca_features_rem)

pca_features_rgb = np.zeros((4 * patch_h * patch_w, 3))
pca_features_rgb[pca_features_bg] = 0
pca_features_rgb[pca_features_fg] = pca_features_rem

pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)
pca_features_rgb = (pca_features_rgb*255).astype(np.uint8)

pca_features_rgb = pca_features_rgb.reshape(2, 2, patch_h, patch_w, 3)
pca_features_rgb = pca_features_rgb.transpose((0, 2, 1, 3, 4))
pca_features_rgb = pca_features_rgb.reshape(2*patch_h, 2*patch_w, 3)

pca_features_rgb = cv2.resize(pca_features_rgb, (512, 512))

inputs: np.ndarray = np.stack([cv2.resize(cv2.imread(f"/root/crane{i+1}.jpg"), (256, 256)) for i in range(4)])
inputs = inputs.reshape(2, 2, 256, 256, 3)
inputs = inputs.transpose((0, 2, 1, 3, 4))
inputs = inputs.reshape(2*256, 2*256, 3)

assert cv2.imwrite(f"result.png", np.hstack([inputs, pca_features_rgb]))
