"""Initial camera of a generated scene in the raw GS (export) frame, for a web splat viewer."""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import nav_env
env = nav_env.Env(sys.argv[1])
c = env.pose(0.0, 0.0, 0.0, 0.0)                      # OpenCV camera-to-world: columns right, down, forward
m_per_export_unit = env.mpu / env.scale
print(json.dumps({'position': c[:3, 3].tolist(), 'forward': c[:3, 2].tolist(), 'up': (-c[:3, 1]).tolist(),
                  'vfov_deg': 60.0, 'm_per_export_unit': m_per_export_unit, 'eye_height_m': nav_env.EYE_M}))
