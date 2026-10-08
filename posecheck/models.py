"""Model adapters. Scores are retained with their original semantics."""
from pathlib import Path
import hashlib
import importlib.util
import cv2
import numpy as np
from .schema import Pose

ROOT = Path(__file__).resolve().parents[1]
RTMW_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip'
DETECTOR_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip'

class Detector:
    def __init__(self):
        from rtmlib import YOLOX
        self.model = YOLOX(DETECTOR_URL, model_input_size=(416,416),
                           score_thr=0.35, backend='onnxruntime',device='cpu')
        self.metadata={'name':'YOLOX-tiny HumanArt','url':DETECTOR_URL}

    def __call__(self,frame):
        h,w=frame.shape[:2]
        factor=min(1.0,960/w)
        image=cv2.resize(frame,(round(w*factor),round(h*factor))) if factor<1 else frame
        boxes=np.asarray(self.model(image),dtype=float).reshape(-1,4)/factor
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
        return boxes

class RTMW:
    def __init__(self,flip_test=True):
        from rtmlib import RTMPose
        self.model = RTMPose(RTMW_URL,model_input_size=(192,256),
                            backend='onnxruntime',device='cpu',to_openpose=False)
        self.metadata={'name':'RTMW-l / Cocktail14 / 256x192', 'source':RTMW_URL,
                       'backend':'ONNX Runtime CPU', 'keypoints':133,
                       'score_kind':'SimCC peak (not calibrated visibility)',
                       'flip_test':False,
                       'note':'Official OpenMMLab ONNX release; not the akore RTMW-m checkpoint.'}

    def predict(self,frame,box):
        xy,score=self.model(frame,bboxes=np.asarray([box],dtype=np.float32))
        return Pose(xy[0,:23].astype(float),score[0,:23].astype(float),
                    np.full(23,np.nan),np.full(23,np.nan),'simcc_peak')

def _verified_probpose_checkpoint():
    from huggingface_hub import hf_hub_download
    path=Path(hf_hub_download('vrg-prague/ProbPose-s','ProbPose-s.pth',
                              revision='main',local_dir=str(ROOT/'models/probpose')))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!='54c4e8cf64ea58c38f7ef49d19e7c0cd6cf6e4de567e23dc1c0ab607122e16f6':
        raise ValueError('ProbPose 가중치 SHA-256이 검토한 공식 파일과 다릅니다. 자동 로드를 중단합니다.')
    return path,digest

class ProbPose:
    def __init__(self,flip_test=True):
        import torch
        from torch import nn
        from sparsemax import Sparsemax
        from mmpretrain.models import VisionTransformer
        torch.set_num_threads(4)
        # mmpretrain provides the original backbone. Head inference mirrors the
        # upstream ProbMapHead; training-only MMPose/MMCV compiled ops are unused.
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv_layers=nn.Sequential(
                    nn.ConvTranspose2d(384,256,4,2,1,bias=False),nn.BatchNorm2d(256),nn.ReLU(),
                    nn.ConvTranspose2d(256,256,4,2,1,bias=False),nn.BatchNorm2d(256),nn.ReLU())
                self.final_layer=nn.Conv2d(256,17,1)
                self.normalize_layer=Sparsemax(dim=-1)
                for name in ('probability','visibility','oks','error'):
                    layers=[]
                    for kernel in ((4,3),(2,2),(2,2)):
                        layers.extend([nn.Conv2d(384,384,3,padding=1),nn.BatchNorm2d(384),
                                       nn.MaxPool2d(kernel,stride=kernel),nn.ReLU()])
                    layers.extend([nn.Conv2d(384,17,1),nn.ReLU() if name=='error' else nn.Sigmoid()])
                    setattr(self,name+'_layers',nn.Sequential(*layers))
            def forward(self,x):
                heat=self.final_layer(self.deconv_layers(x))
                shape=heat.shape
                heat=self.normalize_layer(heat.flatten(2)/0.5).clamp(0,1).reshape(shape)
                return (heat,self.probability_layers(x),self.visibility_layers(x),
                        self.oks_layers(x),self.error_layers(x))
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=VisionTransformer(
                    arch={'embed_dims':384,'num_layers':12,'num_heads':12,'feedforward_channels':1536},
                    img_size=(256,192),patch_size=16,qkv_bias=True,drop_path_rate=0.1,
                    with_cls_token=False,out_type='featmap',patch_cfg={'padding':2},init_cfg=None)
                self.head=Head()
            def forward(self,x):
                return self.head(self.backbone(x)[-1])
        path,digest=_verified_probpose_checkpoint()
        self.model=Net()
        # Trusted upstream checkpoint, byte-for-byte SHA verified before pickle.
        state=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
        self.model.load_state_dict(state,strict=True)
        self.model.eval()
        self.flip_test=flip_test
        self.device='cpu'
        spec=importlib.util.spec_from_file_location('probpose_post',ROOT/'third_party/probpose_post_processing.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.decode=module.get_heatmap_expected_value
        self.metadata={'name':'ProbPose-s','source':'https://huggingface.co/vrg-prague/ProbPose-s',
                       'sha256':digest,'backend':'PyTorch CPU / official mmpretrain backbone',
                       'decoder':'upstream ProbMap expected-OKS decoder', 'keypoints':17,
                       'flip_test':flip_test, 'score_kind':'predicted OKS quality; presence and visibility separate',
                       'adapter':'Portable inference head adapted from ProbMapHead; strict weight loading.'}

    def predict(self,frame,box):
        import torch
        box=np.asarray(box,dtype=np.float32)
        center=(box[:2]+box[2:])/2
        scale=(box[2:]-box[:2])*1.25
        aspect=192/256
        if scale[0]>scale[1]*aspect:scale[1]=scale[0]/aspect
        else:scale[0]=scale[1]*aspect
        sx=191/scale[0];sy=255/scale[1]
        warp=np.array([[sx,0,sx*(-center[0]+scale[0]/2)],
                       [0,sy,sy*(-center[1]+scale[1]/2)]],dtype=np.float32)
        patch=cv2.warpAffine(frame,warp,(192,256),flags=cv2.INTER_LINEAR)
        rgb=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb=(rgb-np.array([123.675,116.28,103.53],dtype=np.float32))/np.array([58.395,57.12,57.375],dtype=np.float32)
        tensor=torch.from_numpy(rgb.transpose(2,0,1)[None])
        with torch.inference_mode():
            outputs=self.model(tensor)
            if self.flip_test:
                flipped=self.model(tensor.flip(-1))
                order=[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
                outputs=tuple((a+(b[:,order].flip(-1) if j==0 else b[:,order]))*0.5
                              for j,(a,b) in enumerate(zip(outputs,flipped)))
        heat,prob,vis,quality,error=[x.cpu().numpy() for x in outputs]
        point,_=self.decode(heat[0])
        # Official ProbMap decoding + TopdownPoseEstimator coordinate restoration.
        point=point/np.array([47,63])*scale+center-scale/2
        result=Pose.empty()
        result.xy[:17]=point
        result.score[:17]=quality.flatten()
        result.presence[:17]=prob.flatten()
        result.visibility[:17]=vis.flatten()
        result.score_kind='predicted_oks'
        return result

def load_model(name,flip_test=True):
    if name=='ProbPose-s':return ProbPose(flip_test)
    if name=='RTMW-l':return RTMW(flip_test)
    raise ValueError(name)
