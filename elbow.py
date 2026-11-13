import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ===== 0. 画像読み込み =====
bgr_image = cv2.imread('page_9.png')

# ===== 1. 先に赤線を消す =====
hsv_tmp = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
h0, s0, v0 = hsv_tmp[:, :, 0], hsv_tmp[:, :, 1], hsv_tmp[:, :, 2]
red_mask = ((h0 <= 10) | (h0 >= 170)) & (s0 >= 80)
clean = bgr_image.copy()
clean[red_mask] = (255, 255, 255)   # 赤を白で塗りつぶす

# ===== 2. 赤を消した画像から HSV を取り直す =====
hsv_image = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

# ===== 3. S・V を特徴量にして標準化 =====
features = np.stack([s.ravel(), v.ravel()], axis=1).astype(np.float32)
mean = features.mean(axis=0)
std  = features.std(axis=0)
features_scaled = (features - mean) / std

# ===== 4. エルボー法で k を検討（速度のため間引いて計算） =====
sub = features_scaled[np.random.choice(len(features_scaled), 50000, replace=False)]
inertias = []
K_range = range(1, 11)
for kk in K_range:
    km_tmp = KMeans(n_clusters=kk, random_state=0, n_init=10)
    km_tmp.fit(sub)
    inertias.append(km_tmp.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(list(K_range), inertias, marker='o')
plt.xlabel('number of clusters k')
plt.ylabel('inertia (within-cluster SSE)')
plt.title('Elbow method')
plt.show()

# ===== 5. 本番の k-means（エルボーを見て k を決める） =====
k = 4   # ← エルボーの折れ曲がりを見て設定し直す
km = KMeans(n_clusters=k, random_state=0, n_init=10)
labels = km.fit_predict(features_scaled)
label_img = labels.reshape(s.shape)

# ===== 6. クラスタごとに表示（どれが鉛筆か確認） =====
fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
if k == 1: axes = [axes]
for c in range(k):
    m = (label_img == c)
    axes[c].imshow(m, cmap='gray')
    axes[c].set_title(f"cluster {c}\nS={s[m].mean():.0f}, V={v[m].mean():.0f}")
    axes[c].axis('off')
plt.tight_layout(); plt.show()

# ===== 7. 鉛筆クラスタ（Vが最小）を取り出す =====
mean_v = [v.ravel()[labels == c].mean() for c in range(k)]
pencil_cluster = int(np.argmin(mean_v))
pencil_mask = (label_img == pencil_cluster)

plt.imshow(pencil_mask, cmap='gray')
plt.title(f"pencil cluster = {pencil_cluster}")
plt.axis('off'); plt.show()