"""Download official MP3D matterport_mesh.zip for the scans used by the MVC benchmark (read-only source)."""
import json, zipfile, requests
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; DST = ROOT / 'outputs/mp3d_dollhouse_20261005'
scans = sorted({v['scan'] for v in json.load(open(HERE / 'scope.json'))['resources'].values()})
for s in scans:
    if list((DST / s).glob(f'{s}/matterport_mesh/*/*.obj')):
        continue
    (DST / s).mkdir(parents=True, exist_ok=True); z = DST / s / 'matterport_mesh.zip'
    for attempt in range(5):
        try:
            with requests.get(f'https://kaldir.vc.in.tum.de/matterport/v1/scans/{s}/matterport_mesh.zip', stream=True, timeout=(20, 120)) as r:
                r.raise_for_status()
                with open(z, 'wb') as f:
                    for ch in r.iter_content(1 << 20): f.write(ch)
            zipfile.ZipFile(z).extractall(DST / s); break
        except Exception as e:
            print('retry', s, type(e).__name__, flush=True)
    print('ok', s, z.stat().st_size // 2**20, 'MB', flush=True)
print('done', len(scans))
