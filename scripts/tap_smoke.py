import sys,time,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/tapnet'))
import torch,cv2
from tapnet.torch.tapir_model import TAPIR
from posecheck.video import probe,frame_at
m=TAPIR(pyramid_level=1,use_casual_conv=True).eval()
m.load_state_dict(torch.load('models/causal_bootstapir_checkpoint.pt',map_location='cpu',weights_only=True),strict=True)
torch.set_num_threads(4)
meta=probe(os.environ['POSE_TEST_VIDEO'])
with torch.inference_mode():
 v=[]
 for t in [float(os.environ.get('POSE_TEST_START','0'))+i*.1 for i in range(3)]:
  f=frame_at(meta,t)[2];v.append(torch.from_numpy(cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),(256,256))).float()[None,None]/127.5-1)
 q=torch.tensor([[[0.,140.,140.],[0.,170.,155.],[0.,110.,135.]]])
 fg=m.get_feature_grids(v[0],False)
 query=m.get_query_features(v[0],False,q,fg)
 state=m.construct_initial_causal_state(3,len(query.resolutions)-1)
 for f in v:
  t=time.perf_counter();fg=m.get_feature_grids(f,False)
  r=m.estimate_trajectories((256,256),False,fg,query,None,64,state,True)
  state=r['causal_context']
  print(time.perf_counter()-t,r['tracks'][-1].tolist(),len(state),flush=True)
print('OK',flush=True)
