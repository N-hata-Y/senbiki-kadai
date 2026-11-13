"""黄色マーカー(7個)を基準に、記入済み画像を基準画像(original)へ位置合わせする。

出力 (output_dir 以下):
    aligned/<name>.png   基準画像の座標系へ変換した画像
    overlay/<name>.png   重ね合わせ確認 (赤=基準画像の線, シアン=位置合わせ後の線, 黒=一致)
    markers/<name>.png   マーカー検出結果 (番号は基準画像との対応)
    transforms.csv       各画像の変換行列と残差 RMS
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MarkerConfig:
    num_markers: int
    detect_scale: float
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    min_area: int
    max_area: int
    max_aspect_diff: float
    dot_radius: int
    search_radius: int
    min_dot_score: float


# ---------------------------------------------------------------- 検出
def _yellow_mask(img: np.ndarray, cfg: MarkerConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def detect_markers(img: np.ndarray, cfg: MarkerConfig) -> np.ndarray:
    """黄色マーカーの中心 (中央の黒点の位置, x, y) を元画像の座標で返す。shape = (num_markers, 2)"""
    s = cfg.detect_scale
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    n, _, stats, _ = cv2.connectedComponentsWithStats(_yellow_mask(small, cfg))

    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (cfg.min_area <= area <= cfg.max_area):
            continue
        if abs(w / h - 1.0) > cfg.max_aspect_diff:
            continue
        cands.append((area, x, y, w, h))

    if len(cands) < cfg.num_markers:
        raise RuntimeError(
            f"マーカーが {len(cands)} 個しか見つかりません (必要: {cfg.num_markers})"
        )
    cands = sorted(cands, reverse=True)[: cfg.num_markers]

    # 黄色領域の中心はマーカーの陰影の影響で数十px ずれるため、
    # 元解像度でマーカー中央の黒点を探して正確な中心とする
    pts = []
    for _, x, y, w, h in cands:
        cx, cy = (x + w / 2) / s, (y + h / 2) / s
        pts.append(_find_center_dot(img, cx, cy, cfg))
    return np.array(pts, dtype=np.float64)


def _dot_template(radius: int) -> np.ndarray:
    """白地に黒い円盤のテンプレート"""
    margin = int(radius * 0.8)
    size = 2 * (radius + margin) + 1
    t = np.full((size, size), 255, np.uint8)
    cv2.circle(t, (size // 2, size // 2), radius, 0, -1)
    return t.astype(np.float32)


def _subpixel_peak(res: np.ndarray, px: int, py: int) -> tuple[float, float]:
    """相関マップのピーク位置を2次関数補間でサブピクセル化"""
    def offset(a: float, b: float, c: float) -> float:
        d = a - 2 * b + c
        return 0.5 * (a - c) / d if d != 0 else 0.0

    dx = offset(*res[py, px - 1:px + 2]) if 0 < px < res.shape[1] - 1 else 0.0
    dy = offset(*res[py - 1:py + 2, px]) if 0 < py < res.shape[0] - 1 else 0.0
    return px + dx, py + dy


def _find_center_dot(img: np.ndarray, cx: float, cy: float,
                     cfg: MarkerConfig) -> tuple[float, float]:
    """(cx, cy) 周辺で黒い円盤 (マーカー中央の黒点) を正規化相互相関で探す。
    明るさの違いや、黒点を通る鉛筆・ペンの線に強い。"""
    tmpl = _dot_template(cfg.dot_radius)
    half = tmpl.shape[0] // 2
    r = cfg.search_radius
    H, W = img.shape[:2]
    x0, y0 = max(int(cx - r), 0), max(int(cy - r), 0)
    x1, y1 = min(int(cx + r), W), min(int(cy + r), H)
    roi = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (px, py) = cv2.minMaxLoc(res)
    if score < cfg.min_dot_score:
        raise RuntimeError(f"マーカー ({cx:.0f}, {cy:.0f}) の中央の黒点が見つかりません "
                           f"(一致度 {score:.2f} < {cfg.min_dot_score})")
    sx, sy = _subpixel_peak(res, px, py)
    return x0 + sx + half, y0 + sy + half


# ---------------------------------------------------------------- 対応付け
def _normalize(pts: np.ndarray) -> np.ndarray:
    c = pts - pts.mean(axis=0)
    return c / np.sqrt((c**2).sum(axis=1).mean())


def match_markers(ref: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """ref[i] に対応する tgt の点を並べ替えて返す。

    重心・スケールで正規化した上で 0/90/180/270 度の回転を試し、
    ハンガリアン法で割り当てたうえで、相似変換の残差が最小の対応を採用する。
    (スキャン位置や拡大率が大きくずれていても対応が取れる)
    """
    r, t = _normalize(ref), _normalize(tgt)
    best_cost, best_idx = np.inf, None
    for k in range(4):
        th = k * np.pi / 2
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        tr = t @ R.T
        cost = np.linalg.norm(r[:, None, :] - tr[None, :, :], axis=2)
        _, col = linear_sum_assignment(cost)
        M, _ = cv2.estimateAffinePartial2D(tgt[col], ref, method=cv2.LMEDS)
        if M is None:
            continue
        res = _rms(_apply(M, tgt[col]), ref)
        if res < best_cost:
            best_cost, best_idx = res, col
    if best_idx is None:
        raise RuntimeError("マーカーの対応付けに失敗しました")
    return tgt[best_idx]


# ---------------------------------------------------------------- 変換
def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if M.shape == (3, 3):
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def estimate_transform(src: np.ndarray, dst: np.ndarray, method: str,
                       robust: bool = False, ransac_thresh: float = 3.0) -> np.ndarray:
    """src (対象画像) → dst (基準画像) の変換。

    robust=False: 全点の最小二乗 (誤差を全マーカーに均等に分散)
    robust=True : RANSAC で外れ点を除外 (検出ミスがあり得る場合)
    """
    s, d = src.astype(np.float32), dst.astype(np.float32)
    # robust=False のときはしきい値を巨大にして全点をインライアとして扱う
    th = ransac_thresh if robust else 1e9
    if method == "affine":
        M, _ = cv2.estimateAffine2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=th)
    elif method == "similarity":
        M, _ = cv2.estimateAffinePartial2D(s, d, method=cv2.RANSAC,
                                           ransacReprojThreshold=th)
    elif method == "homography":
        M, _ = (cv2.findHomography(s, d, cv2.RANSAC, th) if robust
                else cv2.findHomography(s, d, 0))
    else:
        raise ValueError(f"未対応の method: {method}")
    if M is None:
        raise RuntimeError("変換行列を推定できませんでした")
    return M


def warp(img: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if M.shape == (3, 3):
        return cv2.warpPerspective(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
        )
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )


# ---------------------------------------------------------------- 確認画像
def make_overlay(ref: np.ndarray, aligned: np.ndarray, scale: float) -> np.ndarray:
    """赤 = 基準画像のみの暗部, シアン = 位置合わせ画像のみの暗部, 黒 = 両方"""
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g_ali = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    out = np.dstack([g_ref, g_ref, g_ali])  # B, G = 基準, R = 位置合わせ後
    return cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def draw_markers(img: np.ndarray, pts: np.ndarray, scale: float) -> np.ndarray:
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for i, (x, y) in enumerate(pts):
        c = (int(x * scale), int(y * scale))
        cv2.circle(out, c, 25, (255, 0, 255), 3)
        cv2.putText(
            out,
            str(i),
            (c[0] + 28, c[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 255),
            3,
        )
    return out


# ---------------------------------------------------------------- main
def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="黄色マーカーによる位置合わせ")
    parser.add_argument(
        "--config", type=Path, required=True, help="設定ファイル (toml)"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="設定ファイル内の相対パスの基準ディレクトリ",
    )
    args = parser.parse_args()

    conf = load_config(args.config)
    root = args.root
    p = conf["paths"]
    input_dir = root / p["input_dir"]
    out_dir = root / p["output_dir"]
    mcfg = MarkerConfig(
        **{
            **conf["marker"],
            "hsv_lower": tuple(conf["marker"]["hsv_lower"]),
            "hsv_upper": tuple(conf["marker"]["hsv_upper"]),
        }
    )
    method = conf["transform"]["method"]
    max_rms = conf["transform"]["max_rms"]
    robust = conf["transform"].get("robust", False)
    ransac_thresh = conf["transform"].get("ransac_thresh", 3.0)
    o = conf["output"]

    sub = {k: out_dir / k for k in ("aligned", "overlay", "markers")}
    for d in sub.values():
        d.mkdir(parents=True, exist_ok=True)

    # 基準画像
    ref_path = input_dir / p["reference"]
    ref = cv2.imread(str(ref_path))
    if ref is None:
        print(f"基準画像を読み込めません: {ref_path}", file=sys.stderr)
        return 1
    ref_pts = detect_markers(ref, mcfg)
    # 番号を見やすくするため 上→下, 左→右 の順に並べる
    ref_pts = ref_pts[np.lexsort((ref_pts[:, 0], np.round(ref_pts[:, 1] / 300)))]
    h, w = ref.shape[:2]
    print(f"[基準] {ref_path.name}: マーカー {len(ref_pts)} 個検出")
    if o["save_markers"]:
        cv2.imwrite(
            str(sub["markers"] / ref_path.name),
            draw_markers(ref, ref_pts, o["preview_scale"]),
        )

    targets = sorted(
        input_dir.glob(p["targets"]),
        key=lambda q: [
            int(t) if t.isdigit() else t for t in re.split(r"(\d+)", q.stem)
        ],
    )
    if not targets:
        print(f"対象画像が見つかりません: {input_dir / p['targets']}", file=sys.stderr)
        return 1

    rows = []
    n_fail = 0
    for path in targets:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[skip] 読み込めません: {path.name}", file=sys.stderr)
            n_fail += 1
            continue
        try:
            pts = match_markers(ref_pts, detect_markers(img, mcfg))
            M = estimate_transform(pts, ref_pts, method, robust, ransac_thresh)
        except RuntimeError as e:
            print(f"[fail] {path.name}: {e}", file=sys.stderr)
            n_fail += 1
            continue

        res = np.linalg.norm(_apply(M, pts) - ref_pts, axis=1)
        rms = float(np.sqrt((res**2).mean()))
        flag = "  ※残差大" if rms > max_rms else ""
        print(f"[ok]   {path.name}: RMS={rms:.2f}px  max={res.max():.2f}px{flag}")

        aligned = warp(img, M, (w, h))
        if o["save_aligned"]:
            cv2.imwrite(str(sub["aligned"] / path.name), aligned)
        if o["save_overlay"]:
            cv2.imwrite(
                str(sub["overlay"] / path.name),
                make_overlay(ref, aligned, o["preview_scale"]),
            )
        if o["save_markers"]:
            cv2.imwrite(
                str(sub["markers"] / path.name),
                draw_markers(img, pts, o["preview_scale"]),
            )

        Mf = M if M.shape == (3, 3) else np.vstack([M, [0, 0, 1]])
        rows.append(
            [
                path.name,
                method,
                f"{rms:.3f}",
                f"{res.max():.3f}",
                *[f"{v:.8g}" for v in itertools.chain.from_iterable(Mf)],
            ]
        )

    with open(out_dir / "transforms.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "file",
                "method",
                "rms_px",
                "max_px",
                *[f"m{i}{j}" for i in range(3) for j in range(3)],
            ]
        )
        wr.writerows(rows)

    print(f"完了: {len(rows)} 枚成功, {n_fail} 枚失敗 → {out_dir}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
