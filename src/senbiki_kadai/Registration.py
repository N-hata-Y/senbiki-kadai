import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===== 0. 読み込み =====
blank  = cv2.imread('original_1.png')   # 記入前テンプレート
filled = cv2.imread('page_12.png')      # 記入済み
assert blank is not None and filled is not None, "画像が読み込めません"
if blank.shape[:2] != filled.shape[:2]:
    blank = cv2.resize(blank, (filled.shape[1], filled.shape[0]))

# ===== 1. 黄色マーカーの重心を検出 =====
def detect_dots(img, lower=(20, 80, 80), upper=(35, 130, 255), min_area=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))     # 黄色だけ
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    return np.array([cent[i] for i in range(1, n)
                     if stats[i, cv2.CC_STAT_AREA] >= min_area], dtype=np.float32)

pts_blank  = detect_dots(blank)
pts_filled = detect_dots(filled)
print("検出数 blank:", len(pts_blank), " filled:", len(pts_filled))

# ===== 2. 検出点を確認 =====
for img, pts, name in [(blank, pts_blank, "blank"), (filled, pts_filled, "filled")]:
    disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    for (x, y) in pts:
        cv2.circle(disp, (int(x), int(y)), 15, (0, 255, 0), 3)
    plt.figure(figsize=(9, 9)); plt.imshow(disp); plt.title(name); plt.axis('off'); plt.show()

# ===== 3. 最近傍で対応づけ =====
def match_points(src, dst):
    return np.array([dst[np.argmin(np.linalg.norm(dst - p, axis=1))] for p in src],
                    dtype=np.float32)

matched = match_points(pts_blank, pts_filled)

# ===== 4. アフィン変換を推定（RANSAC）＋退化チェック =====
M, inliers = cv2.estimateAffine2D(pts_blank, matched, method=cv2.RANSAC)
assert M is not None and abs(np.linalg.det(M[:, :2])) > 1e-3, "変換が退化。検出を見直す"
print("inlier数:", int(inliers.sum()))

h_img, w_img = filled.shape[:2]
aligned_blank = cv2.warpAffine(blank, M, (w_img, h_img), borderValue=(255, 255, 255))

# ===== 5. 重ね合わせ確認 =====
ga = cv2.cvtColor(aligned_blank, cv2.COLOR_BGR2GRAY)
gf = cv2.cvtColor(filled,        cv2.COLOR_BGR2GRAY)
overlay = np.zeros((h_img, w_img, 3), dtype=np.uint8)
overlay[:, :, 0] = 255 - ga
overlay[:, :, 1] = 255 - gf
plt.figure(figsize=(8, 8)); plt.imshow(overlay)
plt.title("overlay (yellow = matched)"); plt.axis('off'); plt.show()

# ===== 6. 印刷線を消して鉛筆を残す =====
template_lines = (ga < 128).astype(np.uint8)
template_lines = cv2.dilate(template_lines, np.ones((5, 5), np.uint8))
print("template_lines のピクセル数:", int((template_lines > 0).sum()))

result = filled.copy()
result[template_lines > 0] = (255, 255, 255)
plt.figure(figsize=(9, 9))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("template removed (pencil remains)"); plt.axis('off'); plt.show()