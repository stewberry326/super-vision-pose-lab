"""Fetch the reviewed official source revision; do not install JAX/Colab dependencies."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
COMMIT='730cda1c730877cfedbe01bf87fb1cadb78a565d'
p=ROOT/'vendor/tapnet'
if not p.exists():
    p.parent.mkdir(exist_ok=True)
    subprocess.run(['git','clone','https://github.com/google-deepmind/tapnet.git',str(p)],check=True)
    subprocess.run(['git','-C',str(p),'checkout',COMMIT],check=True)
current=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
if current!=COMMIT:raise SystemExit('공식 소스 버전이 검토한 버전과 다릅니다. 자동 변경하지 않았습니다.')
print('TAP source ready:',COMMIT)
