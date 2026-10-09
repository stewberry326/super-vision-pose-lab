"""Causal v3: directional motion, local chain support and explicit reacquisition.

No new coordinates are inferred. These are heuristic acceptance decisions, not
proof of anatomical correctness. Stable but systematically wrong detections can
still pass; reference annotation is required to compare accuracy.
"""
from dataclasses import dataclass
import numpy as np
from .schema import EDGES,NAMES
from .quality_v2 import QualitySettings,finish_quality


@dataclass
class QualityV3Settings(QualitySettings):
    prediction_horizon_seconds: float = .2
    coherent_motion_tolerance_body: float = .03
    coherent_motion_relative_tolerance: float = .35
    stable_segment_change_ratio: float = .25
    recovery_frames: int = 3
    recovery_step_tolerance_body: float = .025
    recovery_min_score: float = .35


def correct_v3(frames,settings,quality=None):
    q=quality or QualityV3Settings()
    if q.recovery_frames<2:raise ValueError('recovery_frames must be >= 2')
    n=len(frames)
    times=np.array([r['time'] for r in frames])
    raw=np.array([r['pose'].xy for r in frames])
    base=np.array([r['reasons'] for r in frames],dtype=object)
    valid=base=='valid';reasons=base.copy()
    scale=np.array([max(np.linalg.norm(r['box'][2:]-r['box'][:2]),40)
                    if r['box'] is not None else np.nan for r in frames])
    motion=np.full((n,23),np.nan);threshold=np.full((n,23),np.nan)
    edge_flag=np.zeros((n,len(EDGES)),bool);edge_rows=[]
    trusted_length=np.full(len(EDGES),np.nan)
    common=np.zeros((n,2));cumulative=np.zeros(2)
    last=np.full(23,-1,int);last_common=np.zeros((23,2))
    last_base=np.full(23,-1,int);reseed=np.zeros(23,bool)
    velocity=np.zeros((23,2));velocity_known=np.zeros(23,bool)
    speed_history=[[] for _ in NAMES];length_history=[[] for _ in EDGES]
    pending=[[] for _ in NAMES]
    coherent=np.zeros((n,23),bool);recovered=np.zeros((n,23),bool)
    prediction_used=np.zeros((n,23),bool);pending_counts=np.zeros((n,23),int)
    neighbors=[[] for _ in NAMES]
    for e,(a,b) in enumerate(EDGES):
        neighbors[a].append((b,e));neighbors[b].append((a,e))
    for i,r in enumerate(frames):
        locked=r['track_state']=='locked' and np.isfinite(scale[i])
        sequence_gap=i>0 and times[i]-times[i-1]>q.comparison_gap_seconds
        if not locked or sequence_gap:
            last[:]=-1;last_common[:]=0;cumulative[:]=0
            last_base[:]=-1;reseed[:]=False
            velocity[:]=0;velocity_known[:]=False
            pending=[[] for _ in NAMES];speed_history=[[] for _ in NAMES]
            length_history=[[] for _ in EDGES];trusted_length[:]=np.nan
        if not locked:continue
        if i and not sequence_gap and frames[i-1]['track_state']=='locked':
            shared=valid[i,5:17]&valid[i-1,5:17]
            if shared.sum()>=4:
                common[i]=np.median(raw[i,5:17][shared]-raw[i-1,5:17][shared],axis=0)
        cumulative+=common[i]
        residual=np.full((23,2),np.nan);step=np.full((23,2),np.nan)
        flag=np.zeros(23,bool);needs_recovery=np.zeros(23,bool)
        for j in range(23):
            speed_history[j]=[(t,v) for t,v in speed_history[j] if times[i]-t<=q.history_seconds]
            if not valid[i,j]:
                pending[j]=[]
                if last_base[j]>=0 and times[i]-times[last_base[j]]>q.comparison_gap_seconds:
                    last[j]=-1;velocity_known[j]=False;reseed[j]=True
                    speed_history[j]=[]
                    for _,e in neighbors[j]:length_history[e]=[];trusted_length[e]=np.nan
                continue
            last_base[j]=i
            prev=last[j]
            if prev<0:
                if reseed[j]:flag[j]=True;needs_recovery[j]=True
                continue
            dt=times[i]-times[prev]
            if dt<=0:flag[j]=True;needs_recovery[j]=True;continue
            if dt>q.comparison_gap_seconds:
                # Elapsed time alone cannot turn a rejected point into an observation.
                flag[j]=True;needs_recovery[j]=True;continue
            size=max(scale[i],scale[prev])
            delta=raw[i,j]-raw[prev,j]-(cumulative-last_common[j])
            expected_delta=np.zeros(2)
            if velocity_known[j] and dt<=q.prediction_horizon_seconds:
                expected_delta=velocity[j]*dt;prediction_used[i,j]=True
            residual[j]=(delta-expected_delta)/size
            motion[i,j]=np.linalg.norm(residual[j])
            speeds=np.array([v for _,v in speed_history[j]])
            adaptive=0.
            if len(speeds)>=q.minimum_history:
                med=np.median(speeds);mad=1.4826*np.median(abs(speeds-med))
                adaptive=(med+q.motion_mad_factor*mad)*dt
            threshold[i,j]=max(q.motion_floor_body,q.minimum_motion_body_per_second*dt,adaptive)
            flag[j]=motion[i,j]>threshold[i,j]
            if i and valid[i-1,j] and 0<times[i]-times[i-1]<=q.comparison_gap_seconds:
                step[j]=(raw[i,j]-raw[i-1,j]-common[i])/size
        for e,(a,b) in enumerate(EDGES):
            length_history[e]=[(t,v) for t,v in length_history[e] if times[i]-t<=q.history_seconds]
            if not(valid[i,a] and valid[i,b]):continue
            history=length_history[e]
            length=np.linalg.norm(raw[i,a]-raw[i,b])/scale[i]
            med=np.median([v for _,v in history]) if history else np.nan
            change=length/med-1 if med>q.minimum_segment_body else np.nan
            anomaly=len(history)>=q.minimum_history and np.isfinite(change) and abs(change)>settings.bone_change_ratio
            edge_flag[i,e]=anomaly
            # Local support requires two independently moving endpoints, matching
            # vectors AND a stable segment; a stationary neighbor is no support.
            if i and valid[i-1,a] and valid[i-1,b] and np.isfinite(step[[a,b]]).all():
                oldlen=np.linalg.norm(raw[i-1,a]-raw[i-1,b])/scale[i-1]
                norms=np.linalg.norm(step[[a,b]],axis=1)
                tolerance=max(q.coherent_motion_tolerance_body,q.coherent_motion_relative_tolerance*max(norms))
                if (not anomaly and oldlen>q.minimum_segment_body
                    and abs(length/oldlen-1)<=q.stable_segment_change_ratio
                    and min(norms)>q.motion_floor_body*.5
                    and min(norms)/max(norms)>.5
                    and np.linalg.norm(step[a]-step[b])<=tolerance):
                    coherent[i,[a,b]]=True
            if anomaly:
                edge_rows.append({'frame_index':r['index'],'time_seconds':times[i],
                    'edge_index':e,'joint_a':NAMES[a],'joint_b':NAMES[b],
                    'current_length_body':length,'recent_median_body':med,
                    'relative_change':change,'history_pairs':len(history),
                    'suspect_endpoint':'','decision':'segment_warning_only'})
        rejected=np.zeros(23,bool)
        for e,(a,b) in enumerate(EDGES):
            if not edge_flag[i,e]:continue
            for suspect,partner in [(a,b),(b,a)]:
                if (flag[suspect] and not coherent[i,suspect]
                    and np.isfinite(motion[i,partner]) and not flag[partner]
                    and motion[i,suspect]>q.endpoint_dominance*max(motion[i,partner],.01)):
                    rejected[suspect]=True
                    for item in edge_rows[::-1]:
                        if item['frame_index']!=r['index']:break
                        if item['edge_index']==e:
                            item['suspect_endpoint']=NAMES[suspect];item['decision']='endpoint_rejected'
        accepted=np.zeros(23,bool)
        for j in range(23):
            if not valid[i,j]:continue
            pending_recovery=bool(pending[j]) and np.isfinite(motion[i,j]) and motion[i,j]>q.motion_floor_body
            suspicious=rejected[j] or needs_recovery[j] or ((flag[j] or pending_recovery) and not coherent[i,j])
            if suspicious:
                # Causal candidate sequence, in translation-compensated image space.
                point=raw[i,j]-cumulative
                recent=pending[j]
                if recent and not(0<times[i]-recent[-1][0]<=q.comparison_gap_seconds):recent=[]
                recent=recent+[(times[i],point.copy())]
                recent=recent[-q.recovery_frames:];pending[j]=recent
                pending_counts[i,j]=len(recent)
                stable=False
                if len(recent)==q.recovery_frames:
                    pts=np.array([p for _,p in recent]);deltas=np.diff(pts,axis=0)/scale[i]
                    dts=np.diff([t for t,_ in recent])
                    # Compare movement using actual time intervals, not frame count.
                    speed=deltas/dts[:,None]
                    stable=np.max(np.linalg.norm(deltas-np.median(speed,axis=0)*dts[:,None],axis=1))<=q.recovery_step_tolerance_body
                support=[]
                for other,e in neighbors[j]:
                    if not valid[i,other]:continue
                    length=np.linalg.norm(raw[i,j]-raw[i,other])/scale[i]
                    plausible=(np.isfinite(trusted_length[e]) and trusted_length[e]>q.minimum_segment_body
                               and abs(length/trusted_length[e]-1)<=settings.bone_change_ratio)
                    # A long *actual missing observation* invalidates stale segment
                    # references. Three consistent visible frames can seed again.
                    # Continuous rejection alone never triggers this reset.
                    if reseed[j] and i>=q.recovery_frames-1:
                        first=i-q.recovery_frames+1
                        if valid[first:i+1,j].all() and valid[first:i+1,other].all():
                            lengths=np.linalg.norm(raw[first:i+1,j]-raw[first:i+1,other],axis=1)/scale[first:i+1]
                            median=np.median(lengths)
                            plausible=(median>q.minimum_segment_body and median<.65
                                and np.max(abs(lengths/median-1))<=q.stable_segment_change_ratio)
                    # A stable neighboring candidate may support reacquisition;
                    # requiring it to be accepted first creates circular lockout.
                    first=i-q.recovery_frames+1
                    neighbor_stable=False
                    if first>=0 and valid[first:i+1,other].all():
                        dt=np.diff(times[first:i+1])
                        ds=np.diff(raw[first:i+1,other],axis=0)-common[first+1:i+1]
                        steps=ds/scale[i]
                        neighbor_stable=(np.all((dt>0)&(dt<=q.comparison_gap_seconds))
                            and np.max(np.linalg.norm(steps-np.median(steps/dt[:,None],axis=0)*dt[:,None],axis=1))<=q.recovery_step_tolerance_body)
                    if plausible and (not flag[other] or neighbor_stable):support.append((other,e))
                # Facial landmarks have no skeleton edges in this app. Their
                # consistency can be checked, but they never enter joint angles.
                if j<5 and stable:support=[(-1,-1)]
                high_score=np.isfinite(r['pose'].score[j]) and r['pose'].score[j]>=max(settings.confidence,q.recovery_min_score)
                if stable and support and high_score and not rejected[j]:
                    recovered[i,j]=True;accepted[j]=True;pending[j]=[];reseed[j]=False
                    velocity_known[j]=False
                elif rejected[j]:reasons[i,j]='outlier_isolated_motion_and_segment'
                elif needs_recovery[j]:reasons[i,j]='uncertain_reacquisition'
                else:reasons[i,j]='uncertain_motion'
            else:accepted[j]=True;pending[j]=[]
            if accepted[j]:
                prev=last[j];dt=times[i]-times[prev] if prev>=0 else np.nan
                if prev>=0 and 0<dt<=q.prediction_horizon_seconds and not recovered[i,j]:
                    current_velocity=(raw[i,j]-raw[prev,j]-(cumulative-last_common[j]))/dt
                    velocity[j]=current_velocity if not velocity_known[j] else .5*velocity[j]+.5*current_velocity
                    velocity_known[j]=True
                    if np.isfinite(motion[i,j]):speed_history[j].append((times[i],motion[i,j]/dt))
                else:velocity_known[j]=False
                last[j]=i;last_common[j]=cumulative.copy()
        # Do not let isolated rejected lengths redefine a plausible segment.
        # Warnings with two accepted endpoints can adapt to legitimate projection.
        for e,(a,b) in enumerate(EDGES):
            if accepted[a] and accepted[b]:
                length=np.linalg.norm(raw[i,a]-raw[i,b])/scale[i]
                length_history[e].append((times[i],length))
                trusted_length[e]=np.median([v for _,v in length_history[e]])
    out=finish_quality(frames,settings,q,raw,base,reasons,edge_flag,motion,threshold,common,edge_rows)
    diag=out['diagnostics'].rename(columns={'v2_reason':'v3_reason'})
    diag['coherent_chain_support']=coherent.reshape(-1)
    diag['prediction_used']=prediction_used.reshape(-1)
    diag['reacquired']=recovered.reshape(-1)
    diag['candidate_streak']=pending_counts.reshape(-1)
    out['diagnostics']=diag
    return out
