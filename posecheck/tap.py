"""Official causal BootsTAPIR adapter. No look-ahead or model substitution."""
from pathlib import Path
import sys,hashlib,time,urllib.request
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/tapnet'
URL='https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt'

def tap_classes():
    if not (VENDOR/'tapnet/torch/tapir_model.py').exists():
        raise RuntimeError('공식 TAP 소스가 없습니다. scripts/setup_tap.py를 실행해주세요.')
    if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
    from tapnet.torch.tapir_model import TAPIR,QueryFeatures,FeatureGrids
    return TAPIR,QueryFeatures,FeatureGrids

class OnlineTAP:
    def __init__(self,device='cpu',resolution=256):
        if resolution not in (256,512):raise ValueError('TAP 입력 크기는 256 또는 512입니다.')
        self.resolution=resolution
        TAPIR,_,_=tap_classes()
        path=ROOT/'models/causal_bootstapir_checkpoint.pt'
        if not path.exists():
            path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.download')
            urllib.request.urlretrieve(URL,tmp);tmp.replace(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!='87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f':raise ValueError('공식 TAP 가중치 SHA-256 불일치')
        self.device=torch.device(device)
        torch.set_num_threads(4)
        self.model=TAPIR(pyramid_level=1,use_casual_conv=True).eval().to(self.device)
        self.model.load_state_dict(torch.load(path,map_location=self.device,weights_only=True),strict=True)
        self.query=None;self.context=None;self.active=np.zeros(17,bool);self.shape=None
        self.metadata={'name':'Online BootsTAPIR','source':URL,'license':'Apache-2.0',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'backend':str(self.device),
            'input_resolution':[resolution,resolution],'causal':True,'points':17,'upstream':'google-deepmind/tapnet'}
        self.last_grids=None;self.last_frame=None

    def tensor(self,frame):
        rgb=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(self.resolution,self.resolution),interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(rgb.copy()).float().to(self.device)[None,None]/127.5-1

    @torch.inference_mode()
    def features(self,frame):
        self.shape=frame.shape[:2];self.last_frame=self.tensor(frame)
        self.last_grids=self.model.get_feature_grids(self.last_frame,False)

    @torch.inference_mode()
    def reset_points(self,xy,mask):
        # Exactly fixed slots; inactive ones get dummy descriptors and are never accepted.
        h,w=self.shape
        xy=np.asarray(xy)[:17];mask=np.asarray(mask)[:17]&np.isfinite(xy).all(axis=1)
        if not mask.any():return
        ids=np.flatnonzero(mask)
        q=np.zeros((1,len(ids),3),np.float32)
        q[0,:,1]=xy[ids,1]*self.resolution/h;q[0,:,2]=xy[ids,0]*self.resolution/w
        feats=self.model.get_query_features(self.last_frame,False,torch.from_numpy(q).to(self.device),self.last_grids)
        if self.query is None:
            _,QueryFeatures,_=tap_classes()
            low=tuple(torch.zeros((1,17,256),device=self.device) for _ in feats.lowres)
            high=tuple(torch.zeros((1,17,128),device=self.device) for _ in feats.hires)
            self.query=QueryFeatures(low,high,feats.resolutions)
            self.context=self.model.construct_initial_causal_state(17,len(feats.resolutions)-1)
            self.context=[{k:v.clone().to(self.device) for k,v in d.items()} for d in self.context]
        # Per-joint descriptor replacement and recurrent-memory reset.
        for dst,src in zip(self.query.lowres,feats.lowres):dst[:,ids]=src
        for dst,src in zip(self.query.hires,feats.hires):dst[:,ids]=src
        for d in self.context:
            for value in d.values():value[:,ids]=0
        self.active[ids]=True

    def invalidate(self):
        self.query=None;self.context=None;self.active[:]=False

    @torch.inference_mode()
    def predict(self):
        xy=np.full((23,2),np.nan);score=np.full(23,np.nan)
        occ=np.full(23,np.nan);unc=np.full(23,np.nan)
        if self.query is None:return xy,score,occ,unc
        result=self.model.estimate_trajectories((self.resolution,self.resolution),False,self.last_grids,self.query,None,64,self.context,True)
        self.context=result['causal_context']
        pts=result['tracks'][-1][0,:,0].cpu().numpy()
        o=torch.sigmoid(result['occlusion'][-1][0,:,0]).cpu().numpy()
        u=torch.sigmoid(result['expected_dist'][-1][0,:,0]).cpu().numpy()
        h,w=self.shape;pts=pts*np.array([w/self.resolution,h/self.resolution])
        xy[:17]=pts;score[:17]=(1-o)*(1-u);occ[:17]=o;unc[:17]=u
        xy[:17][~self.active]=np.nan;score[:17][~self.active]=np.nan
        return xy,score,occ,unc


def choose_joint(det,det_ok,tracked,tap_ok,disagreement,limit,conflicted=False):
    """No confidence averaging across unrelated score meanings."""
    tap_ok=tap_ok and not conflicted
    if det_ok and tap_ok:
        if disagreement<=limit:return tracked,'tracked','agreement',False
        return np.full(2,np.nan),'unavailable','detector_tracker_disagreement',False
    if det_ok:return det,'redetected','tracker_unreliable',True
    if tap_ok:return tracked,'tracked','detector_unreliable',False
    return np.full(2,np.nan),'unavailable','both_unreliable',False
