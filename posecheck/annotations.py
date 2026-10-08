"""Held-out image references, never clinical ground truth or tracker inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .schema import ANGLES,NAMES
from .processing import angle3


def references(run):
    p=Path(run)/'annotations.csv'
    if not p.exists():return pd.DataFrame()
    labels=pd.read_csv(p)
    for key,default in [('role','evaluation'),('split','evaluation'),('certainty','high'),('condition','unspecified')]:
        if key not in labels:labels[key]=default
    # All variants share the same evaluation exclusions to avoid unfair initialization credit.
    seeds=Path(run)/'tap_seeds.json'
    used=set()
    if seeds.exists():used={(int(r['frame_index']),r['joint']) for r in json.loads(seeds.read_text())}
    labels['input_anchor']=[(int(r.frame_index),r.joint) in used for r in labels.itertuples()]
    return labels


def evaluate(folder,split='evaluation'):
    folder=Path(folder);labels=references(folder.parent)
    if labels.empty:return pd.DataFrame(),pd.DataFrame()
    labels=labels[(labels.role=='evaluation') & (labels.split==split) & (~labels.input_anchor)]
    if labels.empty:return pd.DataFrame(),pd.DataFrame()
    coords=pd.read_csv(folder/'coordinates.csv')
    merged=labels.merge(coords,on=['frame_index','joint'],how='left')
    report=json.loads((folder.parent/'report.json').read_text());s=report['settings']
    if folder.name=='ProbPose-TAP':raw_valid=merged.rejection_reason.eq('valid')
    else:
        raw_valid=(merged.track_state.eq('locked') & (merged.score>=s['confidence']))
        raw_valid &= merged.presence.isna() | (merged.presence>=s['presence'])
        raw_valid &= merged.visibility.isna() | (merged.visibility>=s['visibility'])
        raw_valid &= merged.raw_x.between(1,report['source']['width']-1,inclusive='left') & merged.raw_y.between(1,report['source']['height']-1,inclusive='left')
    merged['raw_accepted_prediction']=raw_valid
    merged['unseen_but_raw_accepted']=(~merged.visible.eq(True)) & raw_valid
    diagonal=np.hypot(report['source']['width'],report['source']['height'])
    for variant in ['raw','corrected']:
        pred=raw_valid if variant=='raw' else merged[variant+'_x'].notna() & merged[variant+'_y'].notna()
        valid=merged.visible.eq(True)&pred
        error=np.hypot(merged[variant+'_x']-merged.x,merged[variant+'_y']-merged.y)
        merged[variant+'_error_px']=error.where(valid)
        merged[variant+'_error_image_diagonal']=merged[variant+'_error_px']/diagonal
        merged[variant+'_available']=pred
    manual_angles=[];metric=pd.read_csv(folder/'metrics.csv').set_index('frame_index')
    for frame,group in labels.groupby('frame_index'):
        points={r.joint:[r.x,r.y] for r in group.itertuples() if r.visible and np.isfinite(r.x)}
        for angle,indices in ANGLES.items():
            names=[NAMES[j] for j in indices]
            if all(n in points for n in names) and frame in metric.index:
                reference=angle3(*(points[n] for n in names));row={'frame_index':frame,'angle':angle,'manual_image_angle_deg':reference}
                for variant in ['raw','corrected']:
                    value=metric.loc[frame,angle+'_'+variant+'_deg'];row[variant+'_angle_deg']=value;row[variant+'_absolute_error_deg']=abs(value-reference)
                manual_angles.append(row)
    angles=pd.DataFrame(manual_angles)
    merged.to_csv(folder/f'annotation_errors_{split}.csv',index=False)
    angles.to_csv(folder/f'annotation_angle_errors_{split}.csv',index=False)
    return merged,angles


def compare(run,split='evaluation',large_error_ratio=.02):
    run=Path(run);report=json.loads((run/'report.json').read_text());rows=[];all_errors={}
    for name in report['models']:
        e,_=evaluate(run/name,split)
        if e.empty:continue
        for variant in (['corrected'] if name in ['ProbPose-TAP','ProbPose-v2'] else ['raw','corrected']):
            key=name+' / '+variant
            e=e.copy();e['error']=e[variant+'_error_px'];e['available']=e[variant+'_available']
            all_errors[key]=e
    if not all_errors:return pd.DataFrame(),pd.DataFrame()
    valid_sets=[set(zip(e.loc[e.visible.eq(True)&e.error.notna(),'frame_index'],e.loc[e.visible.eq(True)&e.error.notna(),'joint'])) for e in all_errors.values()]
    common=set.intersection(*valid_sets) if valid_sets else set()
    for key,e in all_errors.items():
        for condition,g in [('all',e),*list(e.groupby('condition'))]:
            visible=g.visible.eq(True);observed=g.loc[visible,'error'].dropna()
            norm=g.loc[visible,'error']/np.hypot(report['source']['width'],report['source']['height'])
            c=g[[((int(r.frame_index),r.joint) in common) for r in g.itertuples()]].error.dropna()
            rows.append({'method':key,'condition':condition,'split':split,'visible_references':int(visible.sum()),
                'accepted_predictions':len(observed),'coverage':len(observed)/max(1,int(visible.sum())),
                'mean_error_px':observed.mean(),'median_error_px':observed.median(),'p95_error_px':observed.quantile(.95),
                'mean_error_image_diagonal':norm.mean(),'large_error_threshold_image_diagonal':large_error_ratio,
                'large_error_rate_among_predictions':float((norm.dropna()>large_error_ratio).mean()) if norm.notna().any() else np.nan,
                'unseen_references':int((~visible).sum()),'unseen_but_accepted_rate':g.loc[~visible,'available'].mean(),
                'common_valid_count':len(c),'common_valid_mean_error_px':c.mean()})
    summary=pd.DataFrame(rows);summary.to_csv(run/f'accuracy_summary_{split}.csv',index=False)
    combined=pd.concat([e.assign(method=k) for k,e in all_errors.items()],ignore_index=True)
    combined.to_csv(run/f'accuracy_details_{split}.csv',index=False)
    return summary,combined


def estimate_lag(reference_times,reference_xy,pred_times,pred_xy,max_gap=.25):
    """Exploratory delay fit; needs dense, moving held-out references, never fills long gaps."""
    rt=np.asarray(reference_times);r=np.asarray(reference_xy);pt=np.asarray(pred_times);p=np.asarray(pred_xy)
    valid=np.isfinite(r).all(-1);rt=rt[valid];r=r[valid]
    if len(rt)<8 or np.ptp(rt)<1:return {'status':'insufficient_dense_references'}
    if np.max(np.diff(rt))>.3:return {'status':'references_too_sparse'}
    if np.max(np.std(r,axis=0))<3:return {'status':'insufficient_motion'}
    best=None
    for lag in np.arange(-.3,.301,.01):
        targets=rt+lag;ix=np.searchsorted(pt,targets,side='right');ok=(ix>0)&(ix<len(pt))
        ids=np.flatnonzero(ok);right=ix[ok];left=right-1;dt=pt[right]-pt[left]
        usable=(dt>0)&(dt<=max_gap)&np.isfinite(p[left]).all(-1)&np.isfinite(p[right]).all(-1)
        ids=ids[usable];left=left[usable];right=right[usable];dt=dt[usable]
        if len(ids)<max(8,int(.8*len(rt))):continue
        f=(targets[ids]-pt[left])/dt
        interp=p[left]*(1-f[:,None])+p[right]*f[:,None]
        rmse=float(np.sqrt(np.mean((interp-r[ids])**2)))
        if best is None or rmse<best['fit_rmse_px']:best={'status':'exploratory_fit','lag_seconds':float(lag),'fit_rmse_px':rmse,'pairs':len(ids),'positive_lag':'prediction follows reference later'}
    return best or {'status':'insufficient_common_samples'}


def delay_report(run,split='evaluation'):
    run=Path(run);labels=references(run);report=json.loads((run/'report.json').read_text());rows=[]
    if labels.empty:return pd.DataFrame()
    labels=labels[(labels.role=='evaluation')&(labels.split==split)&labels.visible.eq(True)&(~labels.input_anchor)]
    for model in report['models']:
        coords=pd.read_csv(run/model/'coordinates.csv')
        for joint,ref in labels.groupby('joint'):
            series=coords[coords.joint==joint].sort_values('time_seconds');ref=ref.merge(series[['frame_index','time_seconds']],on='frame_index').sort_values('time_seconds')
            for variant in (['corrected'] if model in ['ProbPose-TAP','ProbPose-v2'] else ['raw','corrected']):
                values=series[[variant+'_x',variant+'_y']].to_numpy().copy()
                if variant=='raw':
                    # Use the same acceptance semantics as the accuracy comparison.
                    errors,_=evaluate(run/model,split)
                    accepted=set(errors.loc[errors.raw_accepted_prediction & errors.joint.eq(joint),'frame_index'])
                    # For unannotated predictions, raw mask from original model settings.
                    s=report['settings'];mask=series.track_state.eq('locked')&(series.score>=s['confidence'])
                    if model=='ProbPose-TAP':mask=series.rejection_reason.eq('valid')
                    else:
                        mask &= series.presence.isna()|(series.presence>=s['presence'])
                        mask &= series.visibility.isna()|(series.visibility>=s['visibility'])
                    values[~mask]=np.nan
                result=estimate_lag(ref.time_seconds.to_numpy(),ref[['x','y']].to_numpy(),series.time_seconds.to_numpy(),values)
                rows.append({'method':model+' / '+variant,'joint':joint,**result})
    result=pd.DataFrame(rows);result.to_csv(run/f'delay_{split}.csv',index=False);return result
