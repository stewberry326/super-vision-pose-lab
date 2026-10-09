"""Reprocess cached RAW, preserve baseline, build full-interval review queue."""
from pathlib import Path
from datetime import datetime
import json, shutil, tempfile, time
import cv2
import numpy as np
import pandas as pd
from .schema import Settings,Pose,NAMES,LABELS,ANGLES
from .processing import classify,coordinate_table
from .quality_v2 import correct_v2,QualitySettings
from .quality_v3 import correct_v3
from .video import probe,iter_frames
from .pipeline import encode_frames
from .render import overlay,graph_png


def restore_rows(base,model='ProbPose-s'):
    base=Path(base);report=json.loads((base/'report.json').read_text())
    s=Settings(**report['settings']);data=np.load(base/model/'arrays.npz')
    coords=pd.read_csv(base/model/'coordinates.csv')
    rows=[]
    for i,(index,t) in enumerate(zip(data['indices'],data['times'])):
        group=coords[coords.frame_index==index].set_index('joint').reindex(NAMES)
        p=Pose(data['raw'][i].copy(),group.score.to_numpy(),group.presence.to_numpy(),group.visibility.to_numpy(),str(group.score_kind.iloc[0]))
        box=data['boxes'][i];box=None if not np.isfinite(box).all() else box.copy()
        state=str(group.track_state.iloc[0])
        rows.append({'index':int(index),'time':float(t),'box':box,'track_state':state,
            'association_cost':float(group.association_cost.iloc[0]),'pose':p,
            'reasons':classify(p,box,(report['source']['height'],report['source']['width']),s,state)})
    return report,s,data,coords,rows


def review_split(t,start,known_examples=True):
    # The user's previously inspected examples cannot be treated as held-out.
    if known_examples and 17.5<=t<22:return 'calibration'
    return 'evaluation' if int((t-start)//2)%2==0 else 'calibration'


def make_review(old,result,rows,start,known_examples=True,reference=None):
    old_points=old['corrected'];new=result['corrected'];raw=result['raw']
    base=np.array([r['reasons'] for r in rows])
    old_ok=np.isfinite(old_points).all(-1);new_ok=np.isfinite(new).all(-1)
    old_reason=old['reasons'];frame_rows=[]
    for i,r in enumerate(rows):
        body=slice(5,17)
        geometry=(np.isin(old_reason[i],['outlier_bone','outlier_speed']) & (base[i]=='valid'))
        restored=geometry & new_ok[i] & ~old_ok[i]
        removed_now=result['reasons'][i]=='outlier_isolated_motion_and_segment'
        uncertain=result['display_states'][i]=='uncertain'
        vanished=np.zeros(23,bool) if i==0 else old_ok[i-1]&~old_ok[i]&(base[i]=='valid')
        scale=max(np.linalg.norm(r['box'][2:]-r['box'][:2]),40) if r['box'] is not None else np.nan
        displacement=np.linalg.norm(raw[i]-old_points[i],axis=-1)/scale
        discrepancy=(displacement>.015)&old_ok[i]
        flags=[]
        for key,mask in [('old_geometry_removed',geometry),('v2_restored',restored),
            ('v2_rejected',removed_now),('v2_uncertain',uncertain),('old_disappearance',vanished),('raw_old_disagreement',discrepancy)]:
            if mask[body].any():flags.append(key)
        if result['edge_warnings'][i].any():flags.append('segment_warning')
        reference_difference=0
        if reference is not None:
            previous=np.isfinite(reference['corrected'][i]).all(-1)
            delta=np.linalg.norm(reference['corrected'][i]-new[i],axis=-1)/scale
            reference_difference=int(((previous!=new_ok[i]) | (delta>.015))[body].sum())
            if reference_difference:flags.append('v2_v3_disagreement')
        score=int(geometry[body].sum()+restored[body].sum()+3*removed_now[body].sum()+uncertain[body].sum()+2*vanished[body].sum()+discrepancy[body].sum())
        score+=2*reference_difference
        frame_rows.append({'v2_v3_disagreement_points':reference_difference,'sample_index':i,'frame_index':r['index'],'time_seconds':r['time'],
            'candidate':bool(flags),'priority_score':score,'candidate_reasons':' | '.join(flags),
            'old_geometry_removed':int(geometry[body].sum()),'restored_measurement_points':int(restored[body].sum()),
            'v2_rejected_points':int(removed_now[body].sum()),'v2_uncertain_points':int(uncertain[body].sum()),
            'old_display_points':int(old_ok[i,body].sum()),
            'v2_display_points':int(np.isfinite(result['display'][i,body]).all(-1).sum()),
            'v2_measurement_points':int(new_ok[i,body].sum()),
            'suggested_split':review_split(r['time'],start,known_examples),
            'review_status':'pending_original_video_review'})
    all_frames=pd.DataFrame(frame_rows)
    # One risk representative per second, plus systematic and deterministic
    # random samples from the whole interval, including apparently normal ones.
    selected={}
    def add(i,reason):selected.setdefault(int(i),set()).add(reason)
    candidates=all_frames[all_frames.candidate].copy()
    candidates['window']=np.floor(candidates.time_seconds-start).astype(int)
    for _,g in candidates.groupby('window'):add(g.priority_score.idxmax(),'risk_window_peak')
    for t in np.arange(start,rows[-1]['time']+.001,1.):add(abs(all_frames.time_seconds-t).idxmin(),'systematic_sample')
    rng=np.random.default_rng(20261008)
    for i in rng.choice(len(rows),min(10,len(rows)),replace=False):add(i,'fixed_random_sample')
    for t in ([18.2,20.5,21.] if known_examples else []):
        if start<=t<=rows[-1]['time']:add(abs(all_frames.time_seconds-t).idxmin(),'user_reported_example')
    queue=all_frames.loc[sorted(selected)].copy()
    queue['selection_source']=[' | '.join(sorted(selected[i])) for i in queue.index]
    return all_frames,queue.reset_index(drop=True)


def reprocess(base,progress=None,quality=None,version=3):
    before=time.perf_counter();base=Path(base).resolve()
    report,s,old,coords,rows=restore_rows(base)
    known_examples=Path(report['source']['path']).name=='test4_pose.MP4'
    if version not in (2,3):raise ValueError('version must be 2 or 3')
    label=f'v{version}';model=f'ProbPose-{label}'
    result=(correct_v3 if version==3 else correct_v2)(rows,s,quality)
    out=base.parent/datetime.now().strftime('%Y%m%d-%H%M%S-%f');out.mkdir()
    for item in ['ProbPose-s','source_excerpt.mp4','tracking.json','annotations.csv']:
        source=base/item
        if source.is_dir():shutil.copytree(source,out/item)
        elif source.exists():shutil.copy2(source,out/item)
    reference=None
    if version==3 and (base/'ProbPose-v2/arrays.npz').exists():
        shutil.copytree(base/'ProbPose-v2',out/'ProbPose-v2')
        reference=np.load(base/'ProbPose-v2/arrays.npz')
    folder=out/model;folder.mkdir()
    # Preserve existing annotations, but previously inspected frames remain
    # calibration-only in this new experiment.
    if (out/'annotations.csv').exists():
        labels=pd.read_csv(out/'annotations.csv')
        calibration={r['index'] for r in rows if review_split(r['time'],s.start,known_examples)=='calibration'}
        if 'split' not in labels:labels['split']='evaluation'
        labels.loc[labels.frame_index.isin(calibration),'split']='calibration'
        labels.to_csv(out/'annotations.csv',index=False)
    table=result['metrics'];table.to_csv(folder/'metrics.csv',index=False)
    coordinates=coordinate_table(rows,result['raw'],result['corrected'],result['reasons'],result['states'])
    coordinates=coordinates.merge(result['diagnostics'],on=['frame_index','time_seconds','joint'])
    display=result['display'].reshape(-1,2)
    coordinates['display_x']=display[:,0];coordinates['display_y']=display[:,1]
    coordinates.to_csv(folder/'coordinates.csv',index=False)
    result['diagnostics'].to_csv(folder/'quality_diagnostics.csv',index=False)
    result['edge_diagnostics'].to_csv(folder/'segment_warnings.csv',index=False)
    np.savez_compressed(folder/'arrays.npz',raw=result['raw'],corrected=result['corrected'],
        display=result['display'],display_states=result['display_states'].astype(str),
        edge_warnings=result['edge_warnings'],reasons=result['reasons'].astype(str),
        states=result['states'].astype(str),times=old['times'],indices=old['indices'],boxes=old['boxes'])
    frames,queue=make_review(old,result,rows,s.start,known_examples,reference)
    if version==3:
        frames=frames.rename(columns=lambda c:c.replace('v2_','v3_') if not c.startswith('v2_v3_') else c)
        queue=queue.rename(columns=lambda c:c.replace('v2_','v3_') if not c.startswith('v2_v3_') else c)
        for review_table in [frames,queue]:review_table['candidate_reasons']=review_table.candidate_reasons.str.replace('v2_restored','v3_restored',regex=False).str.replace('v2_rejected','v3_rejected',regex=False).str.replace('v2_uncertain','v3_uncertain',regex=False)
    frames.to_csv(out/'review_frames.csv',index=False);queue.to_csv(out/'review_queue.csv',index=False)
    compare_rows=[]
    for j in range(5,17):
        old_available=np.isfinite(old['corrected'][:,j]).all(-1)
        new_available=np.isfinite(result['corrected'][:,j]).all(-1)
        compare_rows.append({'joint':NAMES[j],'label':LABELS[j],
            'base_valid_points':int(sum(r['reasons'][j]=='valid' for r in rows)),
            'old_measurement_points':int(old_available.sum()),'v2_measurement_points':int(new_available.sum()),
            'v2_display_points':int(np.isfinite(result['display'][:,j]).all(-1).sum()),
            'old_outlier_bone':int(np.sum(old['reasons'][:,j]=='outlier_bone')),
            'v2_rejected':int(np.sum(result['reasons'][:,j]=='outlier_isolated_motion_and_segment')),
            'v2_uncertain_motion':int(np.sum(result['reasons'][:,j]=='uncertain_motion'))})
    joint_table=pd.DataFrame(compare_rows)
    if version==3:
        joint_table=joint_table.rename(columns=lambda c:c.replace('v2_','v3_') if not c.startswith('v2_v3_') else c)
        if reference is not None:
            joint_table['v2_measurement_points']=[int(np.isfinite(reference['corrected'][:,j]).all(-1).sum()) for j in range(5,17)]
    joint_table.to_csv(out/'rule_comparison.csv',index=False)
    meta=probe(report['source']['path']);reviewdir=out/'review';reviewdir.mkdir()
    if progress:progress(.1,'새 규칙 및 전체 구간 검토 목록 생성')
    with tempfile.TemporaryDirectory(prefix=f'quality-{label}-') as temp:
        temp=Path(temp)
        for name in ['comparison','corrected','measurement']:(temp/name).mkdir()
        review_ids=set(queue['sample_index'])
        for i,(index,t,frame) in enumerate(iter_frames(meta,old['indices'].tolist())):
            box=rows[i]['box'];state=rows[i]['track_state']
            panels=[overlay(frame,result['raw'][i],np.full(23,'observed'),box,t,state,'ProbPose RAW'),
                overlay(frame,old['corrected'][i],old['states'][i],box,t,state,'Legacy CORRECTED'),
                overlay(frame,result['display'][i],result['display_states'][i],box,t,state,f'Quality {label} DISPLAY',edge_warnings=result['edge_warnings'][i])]
            display_panel=panels[-1]
            if reference is not None:
                panels.insert(2,overlay(frame,reference['display'][i],reference['display_states'][i],box,t,state,'Quality v2 DISPLAY',edge_warnings=reference['edge_warnings'][i]))
                pair=np.vstack([np.hstack([cv2.resize(p,(640,360)) for p in panels[:2]]),np.hstack([cv2.resize(p,(640,360)) for p in panels[2:]])])
            else:pair=np.hstack([cv2.resize(p,(640,360)) for p in panels])
            cv2.imwrite(str(temp/'comparison'/f'{i:06d}.jpg'),pair)
            cv2.imwrite(str(temp/'corrected'/f'{i:06d}.jpg'),display_panel)
            measurement=overlay(frame,result['corrected'][i],result['states'][i],box,t,state,f'Quality {label} MEASUREMENT',edge_warnings=result['edge_warnings'][i])
            cv2.imwrite(str(temp/'measurement'/f'{i:06d}.jpg'),measurement)
            if i in review_ids:
                cv2.imwrite(str(reviewdir/f'{index:06d}_comparison.jpg'),pair)
                cv2.imwrite(str(reviewdir/f'{index:06d}_original.jpg'),cv2.resize(frame,(1280,720)))
            if progress and i%10==0:progress(.1+.75*(i+1)/len(rows),f'전체 구간 비교 영상 {i+1}/{len(rows)}')
        for name in ['comparison','corrected','measurement']:encode_frames(temp/name,old['times'],folder/(name+'.mp4'),s.end)
    shutil.copy2(out/'ProbPose-s/raw.mp4',folder/'raw.mp4')
    graph_png(table,folder/'graphs.png')
    summary={**report['models']['ProbPose-s'],'variant':f'experimental quality-{label}',
        'rejection_counts':coordinates.rejection_reason.value_counts().to_dict(),
        'post_filter_observed_ratio':float(np.mean(result['reasons'][:,:17]=='valid')),
        'interpolation_ratio_body':float(np.mean(result['states'][:,:17]=='interpolated')),
        'display_ratio_body':float(np.isfinite(result['display'][:,:17]).all(-1).mean()),
        'quality_settings':result['settings']}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    previous_summary=report['models'].get('ProbPose-v2')
    report['models']={'ProbPose-s':report['models']['ProbPose-s'],model:summary}
    if reference is not None:report['models']['ProbPose-v2']=previous_summary
    report['models']={k:report['models'][k] for k in ['ProbPose-s','ProbPose-v2','ProbPose-v3'] if k in report['models']}
    report.pop('hybrid',None)
    report['processing_seconds']=time.perf_counter()-before
    report['quality_experiment']={'base_run':str(base),'name':f'causal quality-{label}','version':version,'model':model,
        'comparison_layout':'2x2 RAW / legacy / v2 / v3' if reference is not None else 'RAW / legacy / '+label,
        'reused_raw':True,'inference_rerun':False,'runtime_scope':'postprocessing and rendering only',
        'known_examples':known_examples,
        'parameters':result['settings'],'candidate_frames':int(frames.candidate.sum()),
        'review_queue_frames':len(queue),'accuracy':'pending held-out reference annotations',
        'display_not_measurement':'Yellow uncertain points and segment warnings are diagnostic; warned angles are excluded from metrics.',
        'temporal_scope':'causal quality gate; inherited short-gap interpolation can use future endpoint frames',
        'coverage_scope':f'{s.start}–{s.end} seconds, {len(rows)} sampled frames; not every source frame or the full source video'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    review_summary={'frames':len(rows),'candidate_frames':int(frames.candidate.sum()),'queue_frames':len(queue),
        'old_bone_rejections_body':int(joint_table.old_outlier_bone.sum()),
        'new_combined_rejections_body':int(joint_table[f'{label}_rejected'].sum()),
        'new_uncertain_motion_body':int(joint_table[f'{label}_uncertain_motion'].sum()),
        'accuracy':'not evaluated; review labels not supplied',
        'known_examples_split':'calibration only',
        'rule_success':'not inferred from coverage or smoothness'}
    if version==3:
        review_summary['reacquisition_pending_body']=int(result['diagnostics'].loc[result['diagnostics'].joint.isin(NAMES[5:17])].v3_reason.eq('uncertain_reacquisition').sum())
        review_summary['confirmed_reacquisitions']=int(result['diagnostics'].reacquired.sum())
        review_summary['coherent_chain_support_points']=int(result['diagnostics'].coherent_chain_support.sum())
    (out/'review_summary.json').write_text(json.dumps(review_summary,ensure_ascii=False,indent=2))
    history_note='최근 1초 기준은 경고 길이도 반영합니다.' if version==2 else '최근 분절 이력에는 양 끝점이 통과한 경우만 반영합니다.'
    (out/'보정규칙_검증.md').write_text(f'''# 보정 규칙 {label} 검증

원본 STS {s.start}–{s.end}초의 {len(rows)}개 분석 프레임을 모두 검사했습니다. 전체 원본 {report['source']['duration']:.2f}초/모든 원본 프레임의 검증은 아닙니다.

RAW 검출은 재사용하고 기존 결과를 보존했습니다. 길이 변화만으로 끝점을 제거하지 않습니다. {history_note} 공통 이동을 제외한 관절의 개별 이동과 구간 길이 이상이 함께 있고 한쪽 끝점이 명확히 더 튀는 경우에만 그 점을 제외합니다. 이것도 오류 확정이 아닌 실험적 규칙입니다.

노란 점/점선은 불확실성 표시입니다. 해당 점과 길이 경고가 있는 각도는 측정 결과와 구분합니다. 화면 밖 좌표, 대상 유실, 존재 가능성이 낮은 점은 복원하지 않습니다. 추가 TAP/관절 복원은 적용하지 않았습니다.

문제 후보 {review_summary['candidate_frames']}프레임. 검토 목록 {len(queue)}프레임은 위험 구간 대표, 1초 간격 표본, 고정 난수 표본을 포함합니다. 후보는 정답/오류 라벨이 아닙니다. `review_frames.csv`, `review_queue.csv`, `rule_comparison.csv`, 관절별/연결선별 진단 CSV를 제공합니다.

17.5–22초의 이미 본 사례는 설정 조정용입니다. 다른 구간은 연속 2초 블록으로 용도를 나눴습니다. 같은 영상의 시간 분할은 새로운 환자/영상에 대한 독립 검증이 아닙니다. 최종 평가용 기준점을 보고 규칙을 다시 조정하면 그 점은 더 이상 최종 평가용이 아닙니다.

수동 평가: 예측을 숨긴 원본에서 보이는 점만 지정하고, 가림/잘림을 별도 기록하세요. RAW/기존 보정/{label}의 공통 기준점에서 좌표 오차, 큰 오차 비율, 제공률을 함께 비교합니다. 정상점 제거·오류점 유지 여부는 검토 탭에서 기록합니다. 기준점이 없으므로 정확도 개선은 아직 판단하지 않았습니다.
''')
    if version==3:
        with (out/'보정규칙_검증.md').open('a') as f:f.write('\n\n## v3 추가 변경\n최근 방향을 이용한 짧은 이동 예측, 연결된 관절의 동반 이동 확인, 3개 연속 관측에 의한 재획득을 추가했습니다. 0.4초 경과만으로 의심 좌표를 다시 허용하지 않습니다. 실제 저품질/화면 밖 관측이 길게 이어진 경우 오래된 분절 기준을 초기화하고 다시 확인합니다. 예측은 판정에만 쓰며 관절 좌표를 새로 만들지 않습니다. 기본 짧은 저품질 구간 보간과 평활화는 기존 방식입니다.\n\n일관된 오검출, 옷/가림, 원근 변화는 여전히 실패할 수 있습니다. 제공률 감소가 정확도 향상을 입증하지 않습니다. 수동 기준점 검증 전에는 v2를 대체하는 확정 개선판으로 간주하지 않습니다.\n')
    if progress:progress(1.,f'보정 {label} 비교 및 전체 구간 검토 목록 저장 완료')
    return out
