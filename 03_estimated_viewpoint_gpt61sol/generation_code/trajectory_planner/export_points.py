"""Export a downsampled coloured point cloud of a generated scene in the navigator's metric frame
(initial camera at origin, forward=-X, right=+Y, up=+Z). Binary: uint32 n, float32 xyz[n], uint8 rgb[n]."""
import struct, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import nav_env
env = nav_env.Env(sys.argv[1]); out = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
s = env.splat
keep = np.flatnonzero(s.opacity > .35)
if len(keep) > n:
    keep = np.random.default_rng(0).choice(keep, n, replace=False)
p = s.means[keep].astype(np.float64)
u = ((p - env.t) / env.scale) @ env.R          # export -> original HY units (R orthonormal)
m = (u * env.mpu).astype(np.float32)
sel = (np.abs(m) < 25).all(1); m = m[sel]
rgb = (np.clip(s.color[keep][sel], 0, 1) * 255).astype(np.uint8)
with open(out, 'wb') as f:
    f.write(struct.pack('<I', len(m))); f.write(m.tobytes()); f.write(rgb.tobytes())
print('points', len(m))
