import os, sys, glob
import numpy as np
from plyfile import PlyData

def ply_to_bin(ply_path, out_bin):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    x, y, z = np.asarray(v['x'], dtype=np.float32), np.asarray(v['y'], dtype=np.float32), np.asarray(v['z'], dtype=np.float32)
    intensity = np.zeros_like(x, dtype=np.float32)  # fix: zeros_like (not zeroes_like)
    pts = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    pts.tofile(out_bin)

if __name__ == "__main__":
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for fp in glob.glob(os.path.join(in_dir, "*.ply")):
        bn = os.path.splitext(os.path.basename(fp))[0] + ".bin"
        ply_to_bin(fp, os.path.join(out_dir, bn))
    print("Done.")
