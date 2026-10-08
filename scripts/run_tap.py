import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from posecheck.tap_pipeline import run_tap,TAPSettings
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--device',default='mps');p.add_argument('--interval',type=float,default=.5)
p.add_argument('--resolution',type=int,choices=[256,512],default=256)
a=p.parse_args();last=[-1]
def progress(f,msg):
 n=int(f*40)
 if n!=last[0]:print(msg,flush=True);last[0]=n
print('RESULT',run_tap(a.run,TAPSettings(device=a.device,redetect_seconds=a.interval,resolution=a.resolution),progress),flush=True)
