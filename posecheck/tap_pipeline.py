"""Run a causal hybrid against saved per-frame detector results, preserving A/B."""
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass,asdict
import json,time,shutil,tempfile,resource
import cv2,numpy as np,pandas as pd
from .models import ROOT
from .schema import Pose,Settings,NAMES,ANGLES
from .video import probe,iter_frames
from .tap import OnlineTAP,choose_joint
from .processing import metrics,coordinate_table
from .render import overlay,graph_png
from .pipeline import encode_frames

@dataclass
class TAPSettings:
    redetect_seconds:float=0.5
    visible_threshold:float=0.5
    disagreement_ratio:float=0.06
    detector_confidence:float=0.6
    detector_visibility:float=0.6
    device:str='mps'
    resolution:int=256
    def to_dict(self):return asdict(self)


def run_tap(base_run,config,progress=None,manual_seeds=None):
    base=Path(base_run);original=json.loads((base/'report.json').read_text())
    if 'ProbPose-s' not in original['models']:raise ValueError('ProbPose-s 분석 결과가 필요합니다.')
    s=Settings(**original['settings']);meta=probe(original['source']['path'])
    data=np.load(base/'ProbPose-s/arrays.npz');indices=data['indices'];times=data['times']
    coords=pd.read_csv(base/'ProbPose-s/coordinates.csv')
    locks=json.loads((base/'tracking.json').read_text());manual_seeds=manual_seeds or []
    out=ROOT/'outputs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f');out.mkdir(parents=True)
    # Copy A/B results instead of editing or hardlinking mutable old results.
    for name in ['ProbPose-s','RTMW-l']:
        if (base/name).exists():shutil.copytree(base/name,out/name)
    for name in ['source_excerpt.mp4','tracking.json','annotations.csv']:
        if (base/name).exists():shutil.copy2(base/name,out/name)
    folder=out/'ProbPose-TAP';folder.mkdir()
    started=time.perf_counter();tap=OnlineTAP(config.device,config.resolution);rows=[];extra=[];events=[];timings=[]
    final=[];raw_tap=[];states=[];reasons=[];last_redetect=-np.inf;conflicted=np.zeros(23,bool)
    for i,(index,t,frame) in enumerate(iter_frames(meta,indices.tolist())):
        start=time.perf_counter();lock=locks[i];box=None if lock['box'] is None else np.array(lock['box'])
        result=Pose.empty();pstates=np.full(23,'unavailable',object);why=np.full(23,'unsupported_or_missing',object)
        tapxy=np.full((23,2),np.nan);tscore=np.full(23,np.nan);occ=tscore.copy();unc=tscore.copy()
        detection_due=t-last_redetect>=config.redetect_seconds-1e-6
        at=coords[coords.frame_index==index].set_index('joint').reindex(NAMES)
        detxy=at[['raw_x','raw_y']].to_numpy()
        det_ok=(at.score.to_numpy()>=config.detector_confidence)&(at.presence.to_numpy()>=s.presence)&(at.visibility.to_numpy()>=config.detector_visibility)
        det_ok &= np.isfinite(detxy).all(axis=1)
        h,w=frame.shape[:2];det_ok &= (detxy[:,0]>0)&(detxy[:,0]<w)&(detxy[:,1]>0)&(detxy[:,1]<h)
        reset=np.zeros(23,bool);disagreements=np.full(23,np.nan)
        input_anchor=np.zeros(23,bool);anchor_xy=np.full((23,2),np.nan)
        for anchor in manual_seeds:
            if int(anchor['frame_index'])==index:
                j=NAMES.index(anchor['joint']);anchor_xy[j]=[anchor['x'],anchor['y']]
                if 0<=anchor['x']<w and 0<=anchor['y']<h:input_anchor[j]=True
        if lock['state']!='locked':
            tap.invalidate();conflicted[:]=False;last_redetect=-np.inf;why[:17]='track_'+lock['state']
        else:
            tap.features(frame)
            first=tap.query is None
            # Only current/past predictions are ever used; cached detector is shared for A/B/C.
            if first:
                reset=det_ok.copy();tap.reset_points(detxy,reset)
                tap.reset_points(anchor_xy,input_anchor)
            tapxy,tscore,occ,unc=tap.predict()
            tap_ok=(tscore>=config.visible_threshold)&np.isfinite(tapxy).all(axis=1)
            tap_ok &= (tapxy[:,0]>0)&(tapxy[:,0]<w)&(tapxy[:,1]>0)&(tapxy[:,1]<h)
            # Reject tracks clearly outside the current selected-patient box.
            if box is not None:
                pad=(box[2:]-box[:2])*.15
                tap_ok &= (tapxy[:,0]>=box[0]-pad[0])&(tapxy[:,0]<=box[2]+pad[0])&(tapxy[:,1]>=box[1]-pad[1])&(tapxy[:,1]<=box[3]+pad[1])
            scale=max(np.linalg.norm(box[2:]-box[:2]),40) if box is not None else np.hypot(w,h)
            disagreements=np.linalg.norm(detxy-tapxy,axis=1)/scale
            for j in range(17):
                if input_anchor[j]:
                    result.xy[j]=anchor_xy[j];pstates[j]='manual_anchor';why[j]='manual_initialization';reset[j]=True;conflicted[j]=False
                elif first and det_ok[j]:
                    result.xy[j]=detxy[j];pstates[j]='detection';why[j]='initial_detection';conflicted[j]=False
                elif detection_due:
                    p,state,reason,restart=choose_joint(detxy[j],det_ok[j],tapxy[j],tap_ok[j],disagreements[j],config.disagreement_ratio,conflicted[j])
                    result.xy[j]=p;pstates[j]=state;why[j]=reason;reset[j]=restart
                    if reason=='detector_tracker_disagreement':conflicted[j]=True
                    if restart:conflicted[j]=False
                elif conflicted[j]:why[j]='awaiting_redetection'
                elif tap_ok[j]:result.xy[j]=tapxy[j];pstates[j]='tracked';why[j]='between_detections'
                else:why[j]='track_unreliable'
                if np.isfinite(result.xy[j]).all():result.score[j]=tscore[j] if pstates[j]=='tracked' else at.score.iloc[j]
            if detection_due:
                if not first:tap.reset_points(detxy,reset&~input_anchor)
                last_redetect=t
            if input_anchor.any() and not first:tap.reset_points(anchor_xy,input_anchor)
            for j in np.flatnonzero(reset|input_anchor):
                events.append({'frame_index':index,'time_seconds':t,'joint':NAMES[j],'reason':str(why[j]),
                               'source':'manual' if input_anchor[j] else 'detector'})
        accepted=np.isfinite(result.xy).all(axis=1)
        result.score_kind='selected source; TAP=(1-sigmoid(occlusion))*(1-sigmoid(expected_dist)); detector=predicted OKS'
        rows.append({'index':index,'time':t,'box':box,'track_state':lock['state'],'association_cost':lock['cost'],
                     'pose':result,'reasons':np.where(accepted,'valid',why).tolist()})
        final.append(result.xy.copy());raw_tap.append(tapxy);states.append(pstates);reasons.append(np.where(accepted,'valid',why))
        for j,name in enumerate(NAMES):
            extra.append({'frame_index':index,'time_seconds':t,'joint':name,
                'detection_used':bool(detection_due or (i==0)),'detector_x':detxy[j,0] if detection_due or i==0 else np.nan,
                'detector_y':detxy[j,1] if detection_due or i==0 else np.nan,'detector_score':at.score.iloc[j] if detection_due or i==0 else np.nan,
                'tracker_x':tapxy[j,0],'tracker_y':tapxy[j,1],'tracker_score':tscore[j],'occlusion_probability':occ[j],
                'uncertainty_probability':unc[j],'disagreement_body_ratio':disagreements[j] if detection_due else np.nan,
                'source':pstates[j],'selection_reason':why[j],'reinitialized':bool(reset[j]|input_anchor[j]),
                'evaluation_excluded':bool(input_anchor[j]),'conflict_quarantined':bool(conflicted[j])})
        timings.append(time.perf_counter()-start)
        if progress:progress(.75*(i+1)/len(indices),f'온라인 TAP 추적 {i+1}/{len(indices)} · {t:.1f}초')
    final=np.array(final);states=np.array(states);reasons=np.array(reasons);raw_tap=np.array(raw_tap)
    # C has no hidden interpolation or smoothing. Only selected causal coordinates.
    table=metrics(rows,final,final,states,s);co=coordinate_table(rows,final,final,reasons,states)
    detail=pd.DataFrame(extra);co=co.merge(detail,on=['frame_index','time_seconds','joint']);co.to_csv(folder/'coordinates.csv',index=False)
    table.to_csv(folder/'metrics.csv',index=False);detail.to_csv(folder/'tracking_details.csv',index=False)
    pd.DataFrame(events).to_csv(folder/'reinitializations.csv',index=False)
    (out/'tap_seeds.json').write_text(json.dumps(manual_seeds,indent=2))
    np.savez_compressed(folder/'arrays.npz',raw=final,corrected=final,tracker=raw_tap,reasons=reasons.astype(str),states=states.astype(str),
        times=times,indices=indices,boxes=data['boxes'])
    graph_png(table,folder/'graphs.png')
    with tempfile.TemporaryDirectory(prefix='tap-render-') as temp:
        tmp=Path(temp);(tmp/'comparison').mkdir();(tmp/'corrected').mkdir()
        for i,(index,t,frame) in enumerate(iter_frames(meta,indices.tolist())):
            combined=overlay(frame,final[i],states[i],rows[i]['box'],t,rows[i]['track_state'],'ProbPose + causal TAP')
            baseline=overlay(frame,data['raw'][i],np.full(23,'observed'),rows[i]['box'],t,rows[i]['track_state'],'ProbPose RAW')
            clean=overlay(frame,data['corrected'][i],data['states'][i],rows[i]['box'],t,rows[i]['track_state'],'ProbPose FILTERED')
            pair=np.hstack([cv2.resize(f,(640,360)) for f in [baseline,clean,combined]])
            cv2.imwrite(str(tmp/'comparison'/f'{i:06d}.jpg'),pair);cv2.imwrite(str(tmp/'corrected'/f'{i:06d}.jpg'),combined)
        encode_frames(tmp/'comparison',times,folder/'comparison.mp4',s.end)
        encode_frames(tmp/'corrected',times,folder/'corrected.mp4',s.end)
        shutil.copy2(folder/'corrected.mp4',folder/'raw.mp4')
    rates={name:float(table[name+'_raw_deg'].notna().mean()) for name in [*ANGLES,'trunk']}
    summary={'model':tap.metadata,'frames':len(indices),'raw_valid_rates':rates,'post_filter_observed_ratio':float(np.isfinite(final[:,:17]).all(-1).mean()),
        'interpolation_ratio_body':0.,'track_locked_ratio':float(np.mean([x['state']=='locked' for x in locks])),
        'mean_inference_ms':float(np.mean(timings)*1000),'rejection_counts':co.rejection_reason.value_counts().to_dict(),
        'reinitializations':len(events),'disagreement_events':int(detail.selection_reason.eq('detector_tracker_disagreement').sum()),
        'mode':'causal / no extra smoothing / persistent conflict quarantine','runtime_settings':config.to_dict()}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    report=original.copy();report['models']={k:v for k,v in original['models'].items() if k in ['ProbPose-s','RTMW-l']};report['models']['ProbPose-TAP']=summary
    report['hybrid']={'base_run':str(base),'config':config.to_dict(),'detector_predictions':'cached current-frame predictions for equal A/B/C inputs; hybrid only reads scheduled detections',
        'peak_process_memory_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'manual_seeds':len(manual_seeds),
        'accuracy':'not established without held-out reference annotations','units':'original image pixels; 2D'}
    report['processing_seconds']=time.perf_counter()-started
    report['limitations']=[x for x in original['limitations'] if x!='Core ML and Vision Pro real-time performance have not been tested.']
    report['limitations']+=['The Core ML minimal sample was executed on Mac; Vision Pro device performance has not been tested.',
        'TAP follows surface appearance, not an anatomical joint centre.',
        'Hybrid detector computations are cached; runtime is not end-to-end device latency.',
        'The hybrid did not use future frames; severe disagreement is left unavailable rather than averaged.']
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if progress:progress(1.,'TAP 비교 결과 저장 완료')
    return out
