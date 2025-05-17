import os
from datetime import datetime

import cv2
import numpy as np
import ot
from sklearn.cluster import KMeans

# --- 設定 ---
SOURCE_PATH = "img/cat.jpg"
TARGET_PATH = "img/sunset.jpg"
OUTPUT_DIR  = "output"

# k-means のクラスタ数
N_CLUSTERS = 200

# Sinkhorn の正則化パラメータ（大きめに）
EPS = 0.1

def cluster_lab_centers(img_lab: np.ndarray, k: int):
    H, W = img_lab.shape[:2]
    pts = img_lab.reshape(-1, 3).astype(np.float64)
    km = KMeans(n_clusters=k, random_state=0).fit(pts)
    centers = km.cluster_centers_          # (k,3)
    labels  = km.labels_                   # (H*W,)
    counts  = np.bincount(labels, minlength=k)
    weights = counts.astype(np.float64) / counts.sum()
    return centers, labels, weights, (H, W)

def color_transfer_ot_fixed():
    # 1) 読み込み→LAB変換
    src_bgr = cv2.imread(SOURCE_PATH)
    tgt_bgr = cv2.imread(TARGET_PATH)
    if src_bgr is None or tgt_bgr is None:
        print("Error: 画像の読み込みに失敗しました。パスを確認してください。")
        return
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB)

    # 2) クラスタリング
    src_centers, src_labels, src_w, (H, W) = cluster_lab_centers(src_lab, N_CLUSTERS)
    tgt_centers, tgt_labels, tgt_w, _      = cluster_lab_centers(tgt_lab, N_CLUSTERS)

    # 3) コスト行列計算＆正規化
    C = ot.dist(src_centers, tgt_centers, metric='sqeuclidean')  # (k,k)
    C_max = C.max() or 1.0
    C = C / C_max                                             # [0,1]にスケール

    # 4) Sinkhorn OT
    P = ot.sinkhorn(src_w, tgt_w, C, reg=EPS, numItermax=1000)
    if np.allclose(P, 0):
        print("Warning: Sinkhorn underflow → strict EMD にフォールバック")
        P = ot.emd(src_w, tgt_w, C)

    # 5) マッピング先色を計算
    P_sum = P.sum(axis=1, keepdims=True)
    P_sum[P_sum == 0] = 1.0
    mapped_centers = (P @ tgt_centers) / P_sum
    mapped_centers = np.clip(mapped_centers, 0, 255)

    # --（任意）デバッグ出力
    print("mapped_centers min:", mapped_centers.min(axis=0))
    print("mapped_centers max:", mapped_centers.max(axis=0))

    # 6) 各ピクセルに色を再割当
    out_lab = mapped_centers[src_labels].reshape((H, W, 3)).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    # 7) 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fn = datetime.now().strftime("%Y%m%d_%H%M%S_fixed.jpg")
    out_path = os.path.join(OUTPUT_DIR, fn)
    cv2.imwrite(out_path, out_bgr)
    print(f"▶ カラー・トランスファー完了: {out_path}")

if __name__ == "__main__":
    color_transfer_ot_fixed()
