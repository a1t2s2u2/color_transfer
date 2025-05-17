import cv2
import numpy as np
import ot
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import os

SOURCE_PATH = "img/cat.jpg"
TARGET_PATH = "img/sunset.jpg"

EPS = 0.01
N_ITERS = 200
RESIZE_RATIO = 0.3
SAMPLE_RATIO = 0.3

def load_lab_pixels(path, sample_num=None, resize_ratio=1.0):
    img = cv2.imread(path)
    if resize_ratio < 1.0:
        h0, w0 = img.shape[:2]
        new_h = max(1, int(h0 * resize_ratio))
        new_w = max(1, int(w0 * resize_ratio))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    pts = lab.reshape(-1, 3).astype(np.float32)
    N = pts.shape[0]
    if sample_num is None:
        sample_num = int(N * SAMPLE_RATIO)
    if N > sample_num:
        idx = np.random.choice(N, sample_num, replace=False)
        pts = pts[idx]
    return pts, (h, w)

def main():
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
    mem_bytes = n_pix * sample_num * 4
    mem_mb = mem_bytes / (1024 ** 2)
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"全画素数: {n_pix}, サンプリング数: {sample_num}")
    print(f"想定メモリ消費量: {mem_mb:.1f} MB ({mem_gb:.2f} GB)")
    if mem_gb > 12:
        print("メモリ消費量が12GBを超えるため処理を中断します")
        return

    src_pts, _ = load_lab_pixels(SOURCE_PATH, sample_num=None, resize_ratio=RESIZE_RATIO)
    tgt_pts, _ = load_lab_pixels(TARGET_PATH, sample_num=None, resize_ratio=RESIZE_RATIO)

    N, M = src_pts.shape[0], tgt_pts.shape[0]
    mu = np.ones(N) / N
    nu = np.ones(M) / M

    C = ot.dist(src_pts, tgt_pts, metric='sqeuclidean')
    P = ot.sinkhorn(mu, nu, C, reg=EPS, numItermax=N_ITERS)
    P_sum = np.sum(P, axis=1, keepdims=True)
    P_sum[P_sum == 0] = 1
    src_pts_trans = (P @ tgt_pts) / P_sum

    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(src_pts)
    idx = nn.kneighbors(all_src_pts, return_distance=False).flatten()
    all_trans = src_pts_trans[idx]
    all_trans = np.clip(all_trans, 0, 255)
    lab_out = all_trans.reshape((h, w, 3)).astype(np.uint8)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    os.makedirs("output", exist_ok=True)
    now = datetime.now()
    filename = now.strftime("%Y年%m月%d日_%H時%M分%S秒.jpg")
    out_path = os.path.join("output", filename)
    cv2.imwrite(out_path, bgr_out)
    print(f"▶ カラー・トランスファー完了: {out_path}")

if __name__ == "__main__":
    main()
