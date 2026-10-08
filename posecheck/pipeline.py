from pathlib import Path
from datetime import datetime
import json
import time
import subprocess
import tempfile
import importlib.metadata
import cv2
import numpy as np
from .models import Detector, load_model, ROOT
from .schema import Pose, ANGLES
from .video import probe,sample_indices,iter_frames
from .tracking import TargetLock
from .processing import classify,correct,metrics,coordinate_table
from .render import overlay,graph_png

def encode_frames(directory,times,output,end):
    listing=directory/'frames.ffconcat'
    lines=['ffconcat version 1.0']
    for i,t in enumerate(times):
        # No user-controlled path in concat file; files live in this directory.
        duration=(times[i+1]-t) if i+1<len(times) else max(0.001,end-t)
        lines.extend([f"file '{i:06d}.jpg'",'option framerate 1000',f'duration {duration:.9f}'])
    lines.extend([f"file '{len(times)-1:06d}.jpg'",'option framerate 1000'])
    listing.write_text('\n'.join(lines)+'\n')
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-safe','0','-f','concat','-i',str(listing),
             '-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',
             '-fps_mode','vfr','-video_track_timescale','90000','-t',str(end-times[0]),
             '-movflags','+faststart',str(output)]
    subprocess.run(command,check=True,capture_output=True)

def analyze(path,seed,settings,model_names,progress=None,model_cache=None,detector=None):
    meta=probe(path)
    if not(0<=settings.start<settings.end<=meta['duration']):raise ValueError('분석 시간 범위를 확인해주세요.')
    indices=sample_indices(meta,settings.start,settings.end,settings.sample_fps)
    if len(indices)<2:raise ValueError('분석 구간에 최소 두 프레임이 필요합니다.')
    out=ROOT/'outputs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    started=time.perf_counter();errors={};models={}
    for name in model_names:
        try:
            models[name]=model_cache[name] if model_cache and name in model_cache else load_model(name,settings.flip_test)
        except Exception as exc:
            errors[name]=f'{type(exc).__name__}: {exc}'
    if not models:
        (out/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
        raise RuntimeError('모델을 실행하지 못했습니다: '+json.dumps(errors,ensure_ascii=False))
    detector=detector or Detector()
    lock=TargetLock(seed,settings.max_track_gap)
    records={name:[] for name in models};infer_times={name:[] for name in models}
    locks=[]
    for step,(index,t,frame) in enumerate(iter_frames(meta,indices)):
        boxes=detector(frame)
        box,state,cost=lock.update(frame,boxes,t)
        locks.append({'frame_index':index,'time_seconds':t,'state':state,'cost':None if not np.isfinite(cost) else float(cost),
                      'box':None if box is None else box.tolist(),'detections':boxes.tolist()})
        for name,model in models.items():
            before=time.perf_counter()
            pose=model.predict(frame,box) if box is not None else Pose.empty()
            infer_times[name].append(time.perf_counter()-before)
            records[name].append({'index':index,'time':t,'box':box,'track_state':state,
                                   'association_cost':cost,'pose':pose,
                                   'reasons':classify(pose,box,frame.shape,settings,state)})
        if progress:progress((step+1)/len(indices)*0.65,f'관절 추정 {step+1}/{len(indices)} · {t:.1f}초')
    summaries={}
    for name,rows in records.items():
        folder=out/name;folder.mkdir()
        raw,clean,reasons,states=correct(rows,settings)
        table=metrics(rows,raw,clean,states,settings)
        coordinates=coordinate_table(rows,raw,clean,reasons,states)
        coordinates.to_csv(folder/'coordinates.csv',index=False)
        table.to_csv(folder/'metrics.csv',index=False)
        np.savez_compressed(folder/'arrays.npz',raw=raw,corrected=clean,reasons=reasons.astype(str),
                            states=states.astype(str),times=table.time_seconds.to_numpy(),
                            indices=np.array(indices),boxes=np.array([r['box'] if r['box'] is not None else [np.nan]*4 for r in rows]))
        graph_png(table,folder/'graphs.png')
        lost=sum(r['track_state']!='locked' for r in rows)
        valid_rates={key:float(np.mean([all(r['reasons'][j]=='valid' for j in js) for r in rows])) for key,js in ANGLES.items()}
        valid_rates['trunk']=float(np.mean([all(r['reasons'][j]=='valid' for j in [5,6,11,12]) for r in rows]))
        summaries[name]={'model':models[name].metadata,'frames':len(rows),'raw_valid_rates':valid_rates,
                         'post_filter_observed_ratio':float(np.mean((reasons[:,:17]=='valid'))),
                         'interpolation_ratio_body':float(np.mean(states[:,:17]=='interpolated')),
                         'track_locked_ratio':1-lost/len(rows),
                         'mean_inference_ms':float(np.mean(infer_times[name])*1000),
                         'rejection_counts':coordinates.rejection_reason.value_counts().to_dict()}
        with tempfile.TemporaryDirectory(prefix='pose-render-') as tmp:
            tmp=Path(tmp)
            for variant in ['raw','corrected','comparison']:(tmp/variant).mkdir()
            for step,(index,t,frame) in enumerate(iter_frames(meta,indices)):
                original=raw[step].copy()
                original_states=np.where(np.isfinite(original).all(axis=1),'observed','unavailable')
                raw_image=overlay(frame,original,original_states,rows[step]['box'],t,rows[step]['track_state'],name+' RAW')
                corrected_image=overlay(frame,clean[step],states[step],rows[step]['box'],t,rows[step]['track_state'],name+' CORRECTED')
                cv2.imwrite(str(tmp/'raw'/f'{step:06d}.jpg'),raw_image)
                cv2.imwrite(str(tmp/'corrected'/f'{step:06d}.jpg'),corrected_image)
                # Half-width panels keep the comparison playable on a laptop.
                pair=np.hstack([cv2.resize(raw_image,(640,360)),cv2.resize(corrected_image,(640,360))])
                cv2.imwrite(str(tmp/'comparison'/f'{step:06d}.jpg'),pair)
            for variant in ['raw','corrected','comparison']:
                encode_frames(tmp/variant,table.time_seconds.to_numpy(),folder/(variant+'.mp4'),settings.end)
        (folder/'summary.json').write_text(json.dumps(summaries[name],ensure_ascii=False,indent=2))
        if progress:progress(0.90,f'{name} 결과 영상 저장 완료')
    elapsed=time.perf_counter()-started
    lock_times=np.array([x['time_seconds'] for x in locks]);weights=np.diff(np.r_[lock_times,settings.end])
    versions={k:importlib.metadata.version(k) for k in ['numpy','opencv-contrib-python','torch','rtmlib','onnxruntime']}
    report={'source':{k:v for k,v in meta.items() if k!='timestamps'},'settings':settings.to_dict(),
            'seed_box':list(map(float,seed)),'models':summaries,'model_load_errors':errors,
            'processing_seconds':elapsed,'sampled_frames':len(indices),
            'sampled_frame_throughput':len(indices)/elapsed,
            'track_lost_seconds':float(sum(weight for weight,row in zip(weights,locks) if row['state']!='locked')),
            'association_ambiguities':lock.ambiguous_count,'reacquisitions':lock.reacquisitions,
            'identity_switches':'not objectively measured; inspect video or annotate ground truth',
            'versions':versions,'limitations':[
                '2D image-plane estimates, not validated clinical 3D ROM.',
                'Moving camera, oblique view, overlays and occlusion affect measurements.',
                'Association is a conservative heuristic; identity correctness requires review.',
                'RTMW scores do not establish visibility; in-frame hallucinations remain possible.',
                'Core ML and Vision Pro real-time performance have not been tested.',
                'Angular velocity and onset can be changed by interpolation and smoothing.']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'tracking.json').write_text(json.dumps(locks,ensure_ascii=False,indent=2))
    # Compact original excerpt for playback beside inferred results.
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(settings.start),'-i',meta['path'],
             '-t',str(settings.end-settings.start),'-vf','scale=1280:-2','-an','-c:v','libx264',
             '-preset','veryfast','-crf','23','-movflags','+faststart',str(out/'source_excerpt.mp4')]
    subprocess.run(command,check=True,capture_output=True)
    if progress:progress(1.0,'분석 완료')
    return out
