import cv2
import numpy as np

# ソース画像パス、ターゲット画像パス、出力画像パス
SOURCE_PATH = "img/cat.jpg"
TARGET_PATH = "img/sunset.jpg"
OUTPUT_PATH = "output.jpg"

# Sinkhorn のパラメータ
EPS = 0.01      # エントロピー正則化項 ε
N_ITERS = 200   # 反復回数

def load_lab_pixels(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {path}")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    pts = lab.reshape(-1, 3).astype(np.float64)
    return pts, (h, w)

def sinkhorn(mu, nu, C, eps=EPS, n_iters=N_ITERS):
    # K = exp(-C/ε)
    K = np.exp(-C / eps)
    u = np.ones_like(mu)
    v = np.ones_like(nu)
    for _ in range(n_iters):
        u = mu / (K.dot(v) + 1e-16)
        v = nu / (K.T.dot(u) + 1e-16)
    P = (u[:, None] * K) * v[None, :]
    return P

def main():
    # 1) Lab 空間に変換してベクトル化
    src_pts, (h, w) = load_lab_pixels(SOURCE_PATH)
    tgt_pts, _       = load_lab_pixels(TARGET_PATH)

    N, M = src_pts.shape[0], tgt_pts.shape[0]
    mu = np.ones(N) / N
    nu = np.ones(M) / M

    # 2) コスト行列 C_{ij} = ||x_i - y_j||^2
    diff = src_pts[:, None, :] - tgt_pts[None, :, :]   # (N, M, 3)
    C    = np.sum(diff**2, axis=2)                     # (N, M)

    # 3) Sinkhorn で輸送計画 P を得る
    P = sinkhorn(mu, nu, C, eps=EPS, n_iters=N_ITERS)

    # 4) Barycentric Projection で色マッピング
    src_pts_trans = (P.dot(tgt_pts) / mu[:, None])     # (N, 3)

    # 5) 画素に戻して Lab→BGR 変換、保存
    lab_out = src_pts_trans.reshape((h, w, 3)).astype(np.uint8)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    cv2.imwrite(OUTPUT_PATH, bgr_out)
    print(f"▶ カラー・トランスファー完了: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
