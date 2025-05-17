import cv2
import numpy as np

# ソース画像パス、ターゲット画像パス、出力画像パス
SOURCE_PATH = "img/cat.jpg"
TARGET_PATH = "img/sunset.jpg"
OUTPUT_PATH = "output.jpg"

# Sinkhorn のパラメータ
EPS = 0.01 # エントロピー正則化項 ε
N_ITERS = 200 # 反復回数
RESIZE_RATIO = 0.3 # 圧縮率
SAMPLE_RATIO = 0.3 # サンプリング割合（例: 0.1で10%サンプリング）

def load_lab_pixels(path, sample_num=None, resize_ratio=1.0):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {path}")
    if resize_ratio < 1.0:
        h0, w0 = img.shape[:2]
        new_h = max(1, int(h0 * resize_ratio))
        new_w = max(1, int(w0 * resize_ratio))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    pts = lab.reshape(-1, 3).astype(np.float32)
    # ランダムサンプリング
    N = pts.shape[0]
    if sample_num is None:
        sample_num = int(N * SAMPLE_RATIO)
    if N > sample_num:
        idx = np.random.choice(N, sample_num, replace=False)
        pts = pts[idx]
    return pts, (h, w)

def sinkhorn(mu, nu, C, eps=EPS, n_iters=N_ITERS):
    # K = exp(-C/ε)
    K = np.exp(-C / eps)
    u = np.ones_like(mu)
    v = np.ones_like(nu)
    for _ in range(n_iters):
        u = mu / (K.dot(v) + 1e-16)
        v = nu / (K.T.dot(u) + 1e-16)
    P = (u[:, None] *K) * v[None, :]
    return P

def main():
    # ピクセル数縮小率（例: 0.5で50%サイズ、1.0で縮小なし）

    # 画像読み込み・リサイズ
    img_src = cv2.imread(SOURCE_PATH)
    if RESIZE_RATIO < 1.0:
        h0, w0 = img_src.shape[:2]
        new_h = max(1, int(h0 * RESIZE_RATIO))
        new_w = max(1, int(w0 * RESIZE_RATIO))
        img_src = cv2.resize(img_src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    lab_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
    h, w = lab_src.shape[:2]
    all_src_pts = lab_src.reshape(-1, 3).astype(np.float32)
    n_pix = all_src_pts.shape[0]
    sample_num = int(n_pix * SAMPLE_RATIO)
    mem_bytes = n_pix * sample_num * 4  # float32
    mem_mb = mem_bytes / (1024 ** 2)
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"全画素数: {n_pix}, サンプリング数: {sample_num}")
    print(f"想定メモリ消費量: {mem_mb:.1f} MB ({mem_gb:.2f} GB)")
    if mem_gb > 12:
        print("メモリ消費量が12GBを超えるため処理を中断します")
        return

    # サンプリング点取得
    src_pts, _ = load_lab_pixels(SOURCE_PATH, sample_num=None, resize_ratio=RESIZE_RATIO)
    tgt_pts, _ = load_lab_pixels(TARGET_PATH, sample_num=None, resize_ratio=RESIZE_RATIO)

    N, M = src_pts.shape[0], tgt_pts.shape[0]
    mu = np.ones(N) / N
    nu = np.ones(M) / M

    # 2) コスト行列 C_{ij} = ||x_i - y_j||^2
    diff = src_pts[:, None, :] - tgt_pts[None, :, :]   # (N, M, 3)
    C    = np.sum(diff**2, axis=2)                     # (N, M)

    # 3) Sinkhorn で輸送計画 P を得る
    P = sinkhorn(mu, nu, C, eps=EPS, n_iters=N_ITERS)

    # 4) Barycentric Projection で色マッピング（サンプリング点のみ）
    src_pts_trans = (P.dot(tgt_pts) / mu[:, None])     # (N, 3)

    # 5) 全画素に最近傍割り当て（NumPyのみで実装）
    # all_src_pts: (H*W, 3), src_pts: (N, 3)
    dists = np.sum((all_src_pts[:, None, :] - src_pts[None, :, :]) ** 2, axis=2)  # (H*W, N)
    idx = np.argmin(dists, axis=1)  # (H*W,)
    all_trans = src_pts_trans[idx]
    all_trans = np.clip(all_trans, 0, 255)
    lab_out = all_trans.reshape((h, w, 3)).astype(np.uint8)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    cv2.imwrite(OUTPUT_PATH, bgr_out)
    print(f"▶ カラー・トランスファー完了: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
