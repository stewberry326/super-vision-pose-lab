import argparse
import json
import numpy as np
from posecheck.models import Detector
from posecheck.video import probe,frame_at
from posecheck.schema import Settings
from posecheck.pipeline import analyze

parser=argparse.ArgumentParser(description='Local pose validation on an original video time interval')
parser.add_argument('video')
parser.add_argument('--start',type=float,default=14)
parser.add_argument('--end',type=float,default=34)
parser.add_argument('--fps',type=float,default=10)
parser.add_argument('--seed',type=float,nargs=4,help='Original-pixel x0 y0 x1 y1')
parser.add_argument('--click',type=float,nargs=2,help='Original-pixel point inside the patient at start')
parser.add_argument('--models',nargs='+',default=['ProbPose-s','RTMW-l'])
args=parser.parse_args()
meta=probe(args.video);_,t,frame=frame_at(meta,args.start)
detector=Detector()
seed=args.seed
if seed is None:
    if args.click is None:parser.error('Specify --seed or --click to explicitly select the patient.')
    x,y=args.click;boxes=detector(frame)
    hits=[b for b in boxes if b[0]<=x<=b[2] and b[1]<=y<=b[3]]
    if not hits:parser.error('No detected person at this point. Use --seed or select another frame.')
    seed=min(hits,key=lambda b:np.prod(b[2:]-b[:2]))
settings=Settings(start=t,end=args.end,sample_fps=args.fps)
last=[-1]
def progress(p,msg):
    step=int(p*20)
    if step!=last[0]:print(f'{p:.0%} {msg}',flush=True);last[0]=step
out=analyze(args.video,seed,settings,args.models,progress=progress,detector=detector)
print('RESULT',out,flush=True)
print(json.dumps(json.loads((out/'report.json').read_text()),ensure_ascii=False,indent=2))
