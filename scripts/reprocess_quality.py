import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from posecheck.reprocess import reprocess
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--version',type=int,choices=[2,3],default=3);args=p.parse_args()
print('RESULT',reprocess(args.run,lambda f,msg:print(msg,flush=True),version=args.version))
