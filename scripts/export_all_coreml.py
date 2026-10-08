"""Isolate conversion workers: avoids macOS native-runtime/PyTorch GIL crash."""
from pathlib import Path
import sys,subprocess,json
root=Path(__file__).resolve().parents[1];out=root/'coreml';out.mkdir(exist_ok=True)
report={'units':{},'device_test':'not performed','export_scope':'3 fixed points; 256px; explicit state','notes':[
 'Each unit is converted in a separate process due to a native runtime crash when mixing conversion and multiple loaded models.',
 'Conv1d was lowered to exact three-tap multiply/sum; rank-6 decode was flattened; view_as lowered to fixed reshape.',
 'Mac CPU parity is numerical model parity, not anatomical or clinical accuracy.']}
for unit in ['update','features','query']:
 with (out/f'export-{unit}.log').open('w') as log:
  p=subprocess.run([sys.executable,str(root/'scripts/export_tap_coreml.py'),'--unit',unit],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 file=out/f'report-{unit}.json'
 if file.exists():
  r=json.loads(file.read_text());report.update({k:v for k,v in r.items() if k!='units'});report['units'].update(r['units'])
 if p.returncode!=0:report['units'][unit]={'status':'worker_failed','exit_code':p.returncode,'log':str(out/f'export-{unit}.log')}
 (out/'report.json').write_text(json.dumps(report,indent=2))
 print(unit,report['units'].get(unit),flush=True)
