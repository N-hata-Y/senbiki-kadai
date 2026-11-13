import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs("output", exist_ok=True)

for i in range(1, 17):                       # page_1.png 〜 page_16.png
    img = np.asarray(Image.open(f"page_{i}.png").convert("RGB")).astype(np.float32)

    eps = 1e-6
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    red_ratio = R / (np.maximum(G, B) + eps)
    red_mask = (red_ratio > 1.4) & (R > 60)

    red_result = np.full_like(img, 255)
    red_result[red_mask] = img[red_mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.astype(np.uint8)); axes[0].set_title("Original Image")
    axes[1].imshow(red_mask, cmap='gray'); axes[1].set_title("redMask")
    axes[2].imshow(red_result.astype(np.uint8)); axes[2].set_title("Result Image")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    fig.savefig(f"output/compare_{i}.png", dpi=150)
    plt.close(fig)
    print(f"compare_{i}.png を保存しました")

print("完了")