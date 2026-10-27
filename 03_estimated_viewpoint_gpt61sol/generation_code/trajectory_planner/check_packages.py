"""Format/sanity check of estimated packages against the desktop FO-0003_Data template."""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
SPEC = {'Topdown.png': (1000, 1000), 'Depth.png': (640, 480), 'Normals.png': (640, 480), 'RGB.png': (640, 480), 'Panorama.png': (1024, 512)}
STEP = re.compile(r'^Step \d+: (?:(?:Turn (?:right|left) [\d.]+° in place|Turn around 180\.0° in place|(?:Raise|Lower) your view by [\d.]+° so it is [^.]+)(?: and [^.]+)?\.|Keep your current viewing direction\.) Move (?:forward|backward) [\d.]+ m to the next position, keeping that viewing direction during the movement\. After arriving, (?:keep facing the same direction|.+)\.$')
def check(pkg):
    issues = []
    for f, wh in SPEC.items():
        p = pkg / f
        if not p.exists(): issues.append('missing ' + f); continue
        im = Image.open(p)
        if im.size != wh: issues.append(f'{f} size {im.size}')
        if f in ('RGB.png', 'Panorama.png') and np.asarray(im.convert('L')).std() < 8: issues.append(f'{f} nearly flat (blank render?)')
    v = pkg / 'Video.mp4'
    if not v.exists(): issues.append('missing Video.mp4')
    else:
        o = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt', '-of', 'json', str(v)], capture_output=True, text=True).stdout
        s = json.loads(o)['streams'][0]
        if (s['codec_name'], s['width'], s['height'], s['pix_fmt'], s['r_frame_rate']) != ('h264', 640, 480, 'yuv420p', '12/1'): issues.append('video spec ' + str(s))
        if int(s['nb_read_frames']) < 40: issues.append('short video ' + s['nb_read_frames'])
    t = (pkg / 'Text.txt').read_text(encoding='utf-8').strip().split('\n\n')
    if not t[0].startswith('Begin at the position and viewing direction shown in the starting image. Use that starting direction as 0°.'): issues.append('text header')
    if t[1] != 'Your initial view is level.': issues.append('text initial line')
    steps = [x for x in t if x.startswith('Step ')]
    for x in steps:
        if not STEP.match(x): issues.append('step grammar: ' + x[:90])
    if not t[-1].startswith('Stop here. You have traveled'): issues.append('text stop line')
    look = [x for x in t if x.startswith('At the endpoint,')]
    if pkg.name.startswith('Lookback') and not look: issues.append('lookback without endpoint turn line')
    m = re.search(r'traveled ([\d.]+) m', t[-1]); dist = float(m.group(1)) if m else 0
    if dist < .5: issues.append(f'very short route {dist} m')
    if re.search(r'Turn (?:right|left) 0\.[0-9]°', '\n'.join(t)): issues.append('tiny turn noise')
    return issues
if __name__ == '__main__':
    root = Path(sys.argv[1]); n = 0; bad = 0
    for pkg in sorted(root.glob('*/Estimated_Modalities/*')):
        n += 1; iss = check(pkg)
        if iss: bad += 1; print(pkg.parent.parent.name, pkg.name, iss)
    print(f'{n} packages, {bad} with issues')
