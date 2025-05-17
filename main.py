import os
from datetime import datetime

import cv2
import numpy as np
import ot
from sklearn.cluster import KMeans

# --- 設定: ここで入力画像パスと出力先ディレクトリを指定 ---
SOURCE_PATH = "img/cat.jpg"
TARGET_PATH = "img/sunset.jpg"
OUTPUT_DIR  = "output"

# --- OT + クラスタリング のハイパーパラメータ ---
N_CLUSTERS = 200    # k-means のクラスタ数 (50～500 程度で調整)
EPS        = 0.01   # Sinkhorn の正則化項

def cluster_lab_centers(img_lab: np.ndarray, k: int):
    """
    LAB色空間上の画素を k-means で k クラスタにまとめる。

    Args:
        img_lab: (H,W,3) の LAB 画像
        k:       クラスタ数

    Returns:
        centers: (k,3) 各クラスタの中心色
        labels:  (H*W,) 各画素のクラスタ ID
        weights: (k,) 各クラスタの重み (画素数比)
        (H,W):   元画像の高さ・幅
    """
    H, W = img_lab.shape[:2]
    pts = img_lab.reshape(-1, 3).astype(np.float64)
    km = KMeans(n_clusters=k, random_state=0).fit(pts)
    centers = km.cluster_centers_
    labels  = km.labels_
    counts  = np.bincount(labels, minlength=k)
    weights = counts.astype(np.float64) / counts.sum()
    return centers, labels, weights, (H, W)

def color_transfer_ot():
    # 1) 画像読み込み → LAB に変換
    src_bgr = cv2.imread(SOURCE_PATH)
    tgt_bgr = cv2.imread(TARGET_PATH)
    if src_bgr is None or tgt_bgr is None:
        print("Error: 入力画像の読み込みに失敗しました。パスを確認してください。")
        return
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB)

    # 2) k-means クラスタリング
    src_centers, src_labels, src_w, (H, W) = cluster_lab_centers(src_lab, N_CLUSTERS)
    tgt_centers, tgt_labels, tgt_w, _      = cluster_lab_centers(tgt_lab, N_CLUSTERS)

    # 3) Sinkhorn OT 行列 P を計算
    C = ot.dist(src_centers, tgt_centers, metric='sqeuclidean')  # (k,k) コスト行列
    P = ot.sinkhorn(src_w, tgt_w, C, reg=EPS)                    # (k,k) 乗換行列
    P_sum = P.sum(axis=1, keepdims=True)
    P_sum[P_sum == 0] = 1.0
    # 各ソース中心を移動させた先の色
    mapped_centers = (P @ tgt_centers) / P_sum                  # (k,3)

    # 4) 各画素に対応するクラスタ中心色で出力画像を再構築
    out_lab = mapped_centers[src_labels].reshape((H, W, 3)).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    # 5) ファイル出力
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fn = datetime.now().strftime("%Y%m%d_%H%M%S_ot.jpg")
    out_path = os.path.join(OUTPUT_DIR, fn)
    cv2.imwrite(out_path, out_bgr)
    print(f"▶ カラー・トランスファー完了: {out_path}")

if __name__ == "__main__":
    color_transfer_ot()
