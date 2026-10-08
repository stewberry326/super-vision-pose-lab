import sys,json,time,traceback,warnings,argparse,os
parser=argparse.ArgumentParser();parser.add_argument("--unit",choices=["features","query","update"],required=True);unit=parser.parse_args().unit
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,coremltools as ct
from posecheck.tap import OnlineTAP
from posecheck.tap_export import Features,Query,Update,pack_state,STATE_WIDTH,adapt_temporal_convs,adapt_heatmap_decode
from posecheck.video import probe,frame_at
# Exact fixed-shape lowering for an unsupported reshape alias.
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil import Builder as mb
@register_torch_op
def view_as(context,node):
 x=context[node.inputs[0]];other=context[node.inputs[1]]
 context.add(mb.reshape(x=x,shape=other.shape,name=node.name))

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'coreml';OUT.mkdir(exist_ok=True)
report={'coremltools':ct.__version__,'torch':torch.__version__,'points':3,'input':256,'units':{},'device_test':'not performed'}
torch.set_num_threads(4)
m=OnlineTAP('cpu');report['model']={**m.metadata,'points':3}
video=os.environ.get('POSE_TEST_VIDEO')
if video:
 meta=probe(video)
 start=float(os.environ.get('POSE_TEST_START','0'))
 frames=[m.tensor(frame_at(meta,start+t)[2]) for t in [0,.1,.2]]
else:
 rng=np.random.default_rng(20261008)
 frames=[m.tensor(rng.integers(0,256,(256,256,3),dtype=np.uint8)) for _ in range(3)]
report['parity_input']='provided video' if video else 'synthetic frames'
points=torch.tensor([[[120.,140.],[140.,160.],[150.,190.]]])
with torch.inference_mode():
 fg=m.model.get_feature_grids(frames[0],False,[(256,256)])
 inp=[fg.lowres[0],fg.hires[0]]
 q=Query()(*inp,points)
 qp=torch.cat([torch.zeros((1,3,1)),points.flip(-1)],-1)
 orig=m.model.get_query_features(frames[0],False,qp,fg)
 report['query_adapter_max_error']=float(max((orig.lowres[0]-q[0]).abs().max(),(orig.hires[0]-q[1]).abs().max()))
 mem=pack_state(m.model.construct_initial_causal_state(3,1))
 orig_out=m.model.estimate_trajectories((256,256),False,fg,orig,None,64,m.model.construct_initial_causal_state(3,1),True)
 new_out=Update(m.model)(*inp,*q,mem)
 report['update_adapter_max_error']=float((orig_out['tracks'][-1][:,:,0]-new_out[0]).abs().max())
adapt_temporal_convs(m.model)
adapt_heatmap_decode()
with torch.inference_mode():
 adapted=Update(m.model)(*inp,*q,mem)
 report['temporal_conv_adapter_max_error']=float(max((adapted[j]-new_out[j]).abs().max() for j in range(4)))
units=[('features',Features(m.model),(frames[0],),['frame'],['low','high']),
       ('query',Query(),(*inp,points),['low','high','points'],['query_low','query_high']),
       ('update',Update(m.model),(*inp,*q,mem),['low','high','query_low','query_high','memory'],['tracks','occlusion','uncertainty','next_memory'])]
units=[u for u in units if u[0]==unit]
# Reference sequence is computed before loading Core ML native runtime.
with torch.inference_mode():
 reference_memory=mem;reference=[]
 for i,frame in enumerate(frames):
  fl,fh=Features(m.model)(frame)
  result=Update(m.model)(fl,fh,*q,reference_memory);reference_memory=result[3]
  frame.numpy().astype('float32').tofile(OUT/f'frame-{i}.bin')
  reference.append({'tracks':result[0].numpy().reshape(-1).tolist(),'occlusion':result[1].numpy().reshape(-1).tolist(),'uncertainty':result[2].numpy().reshape(-1).tolist()})
 (OUT/'sequence-reference.json').write_text(json.dumps(reference))
converted={}
for name,wrapper,args,names,outputs in units:
 start=time.perf_counter()
 try:
  with torch.no_grad():
   traced=torch.jit.trace(wrapper.eval(),args,check_trace=False)
  if name!='update':traced.save(str(OUT/(name+'.pt')))
  model=ct.convert(traced,inputs=[ct.TensorType(name=n,shape=tuple(a.shape)) for n,a in zip(names,args)],
        outputs=[ct.TensorType(name=n) for n in outputs],convert_to='mlprogram',minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,compute_units=ct.ComputeUnit.CPU_ONLY)
  model.save(str(OUT/(name+'.mlpackage')))
  with torch.no_grad():expected=wrapper(*args)
  if not isinstance(expected,tuple):expected=(expected,)
  pred=model.predict({n:a.numpy() for n,a in zip(names,args)})
  errors={n:float(np.max(abs(pred[n]-a.numpy()))) for n,a in zip(outputs,expected)}
  report['units'][name]={'status':'converted_and_macos_cpu_executed','max_abs_errors':errors,'seconds':time.perf_counter()-start}
  converted[name]=model
 except Exception as e:
  (OUT/(name+'-error.log')).write_text(traceback.format_exc())
  report['units'][name]={'status':'failed','error':str(e),'seconds':time.perf_counter()-start}
 (OUT/f'report-{unit}.json').write_text(json.dumps(report,indent=2))
 print(name,report['units'][name],flush=True)
if len(converted)==3:
 try:
  ref_mem=mem;cm_mem=mem.numpy();rows=[]
  for frame in frames:
   features=converted['features'].predict({'frame':frame.numpy()});low,high=features['low'],features['high']
   cm=converted['update'].predict({'low':low,'high':high,'query_low':q[0].numpy(),'query_high':q[1].numpy(),'memory':cm_mem})
   with torch.no_grad():
    fl,fh=Features(m.model)(frame);ref=Update(m.model)(fl,fh,*q,ref_mem)
   cm_mem=cm['next_memory'];ref_mem=ref[3]
   rows.append({'track_error_px_256':float(np.max(abs(cm['tracks']-ref[0].numpy()))),'memory_error':float(np.max(abs(cm_mem-ref_mem.numpy())))})
  report['sequence_parity']=rows
 except Exception as e:report['sequence_error']=str(e)
(OUT/f'report-{unit}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
