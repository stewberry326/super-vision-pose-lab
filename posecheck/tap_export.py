"""Fixed 256px, fixed-point causal TAPIR export units; explicit memory tensors."""
import torch
from torch import nn
from .tap import tap_classes

STATE_KEYS=[(f'block_{i}_causal_{j}',512 if j==1 else 2048) for i in range(12) for j in (1,2)]
STATE_WIDTH=sum(x[1] for x in STATE_KEYS)

def pack_state(state):return torch.stack([torch.cat([d[k] for k,_ in STATE_KEYS],-1) for d in state])
def unpack_state(packed):
    result=[]
    for i in range(4):
        offset=0;d={}
        for key,width in STATE_KEYS:d[key]=packed[i,...,offset:offset+width];offset+=width
        result.append(d)
    return result

class Features(nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,frame):
        grids=self.m.get_feature_grids(frame,False,[(256,256)])
        return grids.lowres[0],grids.hires[0]

class Query(nn.Module):
    def forward(self,low,high,points):
        # Upstream map_coordinates_3d at time=0 degenerates to this 2D sample.
        def sample(grid):
            h=grid.shape[2];w=grid.shape[3]
            xy=points*points.new_tensor([w/256,h/256])
            normalized=2*xy/points.new_tensor([w,h])-1
            out=torch.nn.functional.grid_sample(grid[:,0].permute(0,3,1,2),normalized[:,None],
                    mode='bilinear',padding_mode='border',align_corners=False)
            return out[:, :,0].permute(0,2,1)
        return sample(low),sample(high)

class Update(nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,low,high,query_low,query_high,memory):
        # Equivalent fixed batch, single-resolution path without permutation/index_put.
        pts,occ,unc=self.m.tracks_from_cost_volume(query_low,low,None,(1,1,256,256,3))
        pooled=torch.nn.functional.avg_pool2d(low[:,0].permute(0,3,1,2),2,2).permute(0,2,3,1)[:,None]
        pyramid=[high,low,pooled]
        queries=[query_high,query_low,query_low]
        state=unpack_state(memory);new=[];mixer=None
        for i in range(4):
            pts,occ,unc,mixer,s=self.m.refine_pips(queries,None,pyramid,pts,occ,unc,(256,256),
                last_iter=mixer,mixer_iter=i,resize_hw=(256,256),causal_context=state[i],get_causal_context=True)
            new.append(s)
        return pts[:,:,0],torch.sigmoid(occ[:,:,0]),torch.sigmoid(unc[:,:,0]),pack_state(new)

class TemporalThreeTap(nn.Module):
    """Exact depthwise 3-tap sum. Avoids macOS grouped-conv tracing crash."""
    def __init__(self,conv):
        super().__init__()
        assert conv.groups==conv.in_channels and conv.kernel_size==(3,) and conv.padding==(0,)
        self.multiplier=conv.out_channels//conv.in_channels
        self.register_buffer('weight',conv.weight.detach().clone())
        self.register_buffer('bias',conv.bias.detach().clone())
    def forward(self,x):
        n,c,t=x.shape
        expanded=x[:,:,None,:].expand(n,c,self.multiplier,t).reshape(n,c*self.multiplier,t)
        return (expanded[:,:,:-2]*self.weight[:,0,0][None,:,None]
            +expanded[:,:,1:-1]*self.weight[:,0,1][None,:,None]
            +expanded[:,:,2:]*self.weight[:,0,2][None,:,None]+self.bias[None,:,None])

def adapt_temporal_convs(model):
    for block in model.torch_pips_mixer.blocks:
        block.mlp1_up=TemporalThreeTap(block.mlp1_up)
        block.mlp1_up_1=TemporalThreeTap(block.mlp1_up_1)

def adapt_heatmap_decode():
    """Flatten spatial axes before weighting: exact same radius, no rank-6 tensors."""
    from tapnet.torch import utils
    def flat_soft_argmax(value,threshold=5):
        b,n,t,h,w=value.shape
        y,x=torch.meshgrid(torch.arange(h,device=value.device),torch.arange(w,device=value.device),indexing='ij')
        coords=torch.stack([x+.5,y+.5],-1).reshape(h*w,2)
        flat=value.reshape(b*n*t,h*w)
        peak=coords[torch.argmax(flat,-1)]
        delta=coords[None]-peak[:,None]
        valid=(delta.square().sum(-1)<threshold**2)
        weights=flat*valid
        result=(weights[:,:,None]*coords[None]).sum(1)/weights.sum(1).clamp_min(1e-12)[:,None]
        return result.reshape(b,n,t,2)
    utils.soft_argmax_heatmap_batched=flat_soft_argmax
