"""kill any running training python processes (drift / ae / letter bank build)."""
import os, signal, subprocess, sys
keywords = sys.argv[1:] if len(sys.argv) > 1 else ['face_drift', 'face_ae', 'build_letter', 'prep_data']
out = subprocess.check_output(['ps', '-eo', 'pid,args']).decode()
killed = []
for line in out.splitlines()[1:]:
    parts = line.strip().split(None, 1)
    if len(parts) < 2:
        continue
    pid, args = parts
    cmd0 = args.split()[0]
    if cmd0.endswith('python') or cmd0.endswith('python3'):
        if any(k in args for k in keywords):
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed.append((pid, args[:80]))
            except Exception:
                pass
print('killed', killed)
