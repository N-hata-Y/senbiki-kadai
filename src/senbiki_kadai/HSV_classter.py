from sklearn.cluster import KMeans
import numpy as np
import cv2
import matplotlib.pyplot as plt
bgr_image = cv2.imread('page_1.png')
# remove red lines
hsv_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2HSV)
h,s,v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]

# 【追加①】絶対に線でない画素(有彩色 or 真っ白)を除き、候補だけ残す
candidate = ~((s >= 80) | (v >= 245))

#Choose the parameters for KMeans clustering 
features = np.stack([s[candidate], v[candidate]], axis=1).astype(np.float32)  # 【変更②】候補画素だけ

#scailing
mean = features.mean(axis=0)
std  = features.std(axis=0)
features_scaled = (features - mean) / std

#kmeans clustering
k = 2
km = KMeans(n_clusters=k, random_state=0,n_init=10)
labels = km.fit_predict(features_scaled)

# 【変更③】候補だけなので reshape 不可 → 全体サイズに書き戻す（除外は -1）
label_img = np.full(s.shape, -1, dtype=int)
label_img[candidate] = labels

mean_s = [features[km.labels_ == i, 0].mean() for i in range(k)]  # 【変更④】候補の特徴量から
mean_v = [features[km.labels_ == i, 1].mean() for i in range(k)]  # 【変更④】

pencil_cluster = int(np.argmin(mean_v))
pencil_mask = (label_img == pencil_cluster)

# ① 全画素は多すぎるので間引く（例：2万点をランダム抽出）
# N = features.shape[0]
# idx = np.random.choice(N, size=min(20000, N), replace=False)

# plt.figure(figsize=(6, 6))

# #② S(横)・V(縦) に、クラスタ番号で色をつけて散布
# plt.scatter(features[idx, 0], features[idx, 1],
#             c=labels[idx], cmap='viridis', s=2, alpha=0.3)

# ③ クラスタ中心を元のS,Vスケールに戻してプロット
centers = km.cluster_centers_ * std + mean

fig, axes = plt.subplots(1, k, figsize=(5 * k, 5))
if k == 1:
    axes = [axes]   # k=1 のとき axes が配列にならないための保険

for c in range(k):
    cluster_mask = (label_img == c)
    axes[c].imshow(cluster_mask, cmap='gray')   # そのクラスタ=白, 他=黒
    axes[c].set_title(f"cluster {c} (meanV={v[cluster_mask].mean():.0f})")
    axes[c].axis('off')

plt.tight_layout()
plt.show()