"""Wait for the asset builder (PID argv[1]) to exit, then preflight and run the serial evaluation."""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
pid = sys.argv[1]
while pid in subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True).stdout:
    time.sleep(60)
log = open(HERE / 'chain.log', 'a', encoding='utf-8')
log.write(f'assets builder {pid} exited at {time.ctime()}\n'); log.flush()
pre = subprocess.run([sys.executable, '-u', str(HERE / 'run_mvc_modalities.py'), '--preflight'], capture_output=True, text=True)
log.write(pre.stdout + pre.stderr); log.flush()
if '"missing_assets": 0' not in pre.stdout:
    log.write('preflight found missing assets; evaluation NOT started\n'); sys.exit(1)
with open(HERE / 'run.log', 'a', encoding='utf-8') as out, open(HERE / 'run.stderr.log', 'a', encoding='utf-8') as err:
    rc = subprocess.run([sys.executable, '-u', str(HERE / 'run_mvc_modalities.py'), '--execute'], stdout=out, stderr=err).returncode
log.write(f'evaluation exited rc={rc} at {time.ctime()}\n')
