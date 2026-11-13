import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr_image = cv2.imread('page_9.png')
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

# 「絶対に線ではない」= 有彩色(S高) or 真っ白(V極大) を除く
definitely_not = (s >= 80) | (v >= 245)
candidate = ~definitely_not

# 残す画素だけ元の色で、除く画素は白にした画像
kept = bgr_image.copy()
kept[~candidate] = (255, 255, 255)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)); axes[0].set_title("original")
axes[1].imshow(candidate, cmap='gray');                    axes[1].set_title("candidate (white=kept)")
axes[2].imshow(cv2.cvtColor(kept, cv2.COLOR_BGR2RGB));     axes[2].set_title("kept pixels")
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()