import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs("output/black", exist_ok=True)      # output の中に black フォルダを作成

for i in range(2, 3):                            # page_1.png 〜 page_16.png
    img = np.asarray(Image.open(f"page_{i}.png").convert("RGB")).astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    luminance = 0.299*R + 0.587*G + 0.114*B
    max_c = np.maximum(np.maximum(R, G), B)
    min_c = np.minimum(np.minimum(R, G), B)

    black_mask = (luminance <= 255) & (max_c - min_c < 30)

    black_result = np.full_like(img, 0)
    black_result[black_mask] = img[black_mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.astype(np.uint8)); axes[0].set_title("Original Image")
    axes[1].imshow(black_mask, cmap='gray'); axes[1].set_title("blackMask")
    axes[2].imshow(black_result.astype(np.uint8)); axes[2].set_title("Result Image")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    fig.savefig(f"output/black/compare_{i}.png", dpi=150)
    plt.close(fig)
    print(f"black/compare_{i}.png を保存しました")
    np.savetxt(f"test_{i}.csv",luminance,delimiter=",")

print("完了")