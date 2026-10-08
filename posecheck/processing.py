import numpy as np
import pandas as pd
from .schema import NAMES, EDGES, ANGLES

def angle3(a,b,c):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float)
    u=a-b;v=c-b
    norm=np.linalg.norm(u)*np.linalg.norm(v)
    if not np.isfinite(norm) or norm<1e-8:return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/norm,-1,1))))

def classify(pose,box,shape,settings,track_state):
    h,w=shape[:2];reasons=[]
    for j,p in enumerate(pose.xy):
        if track_state!='locked':reason='track_'+track_state
        elif not np.isfinite(p).all():reason='unsupported_or_missing'
        elif not(1<=p[0]<w-1 and 1<=p[1]<h-1):reason='out_of_frame'
        elif np.isfinite(pose.presence[j]) and pose.presence[j]<settings.presence:reason='low_presence'
        elif np.isfinite(pose.visibility[j]) and pose.visibility[j]<settings.visibility:reason='low_visibility'
        elif not np.isfinite(pose.score[j]) or pose.score[j]<settings.confidence:reason='low_score'
        else:reason='valid'
        reasons.append(reason)
    return reasons

def correct(frames,settings):
    """Keep raw coordinates; interpolate only bounded, short, same-target gaps."""
    n=len(frames);times=np.array([r['time'] for r in frames])
    raw=np.array([r['pose'].xy for r in frames])
    reasons=np.array([r['reasons'] for r in frames],dtype=object)
    scales=np.array([max(np.linalg.norm(r['box'][2:]-r['box'][:2]),40)
                     if r['box'] is not None else np.nan for r in frames])
    locked=np.array([r['track_state']=='locked' for r in frames])
    corrected=raw.copy()
    # Spatial jump gate, reset across long gaps. Uses last accepted observation.
    for j in range(23):
        last=None
        for i in range(n):
            if reasons[i,j]!='valid':continue
            if last is not None:
                dt=times[i]-times[last]
                if dt<=0.4:
                    speed=np.linalg.norm(raw[i,j]-raw[last,j])/max(scales[i],scales[last])/dt
                    if speed>settings.max_speed_body_per_second:
                        reasons[i,j]='outlier_speed';continue
            last=i
    # Conservative rolling limb-length consistency gate, normalized by person box.
    for a,b in EDGES:
        history=[]
        for i in range(n):
            if not locked[i]:history=[];continue
            if reasons[i,a]!='valid' or reasons[i,b]!='valid':continue
            length=np.linalg.norm(raw[i,a]-raw[i,b])/scales[i]
            if len(history)>=5:
                median=np.median(history[-10:])
                if median>0.03 and abs(length/median-1)>settings.bone_change_ratio:
                    reasons[i,b]='outlier_bone';continue
            history.append(length)
    valid=reasons=='valid'
    corrected[~valid]=np.nan
    states=np.where(valid,'observed','unavailable').astype(object)
    allowed={'low_score','low_visibility','outlier_speed','outlier_bone'}
    for j in range(23):
        good=np.flatnonzero(valid[:,j])
        for left,right in zip(good[:-1],good[1:]):
            if right-left<2:continue
            if times[right]-times[left]>settings.max_gap_seconds+1e-6:continue
            if not locked[left:right+1].all():continue
            if any(x not in allowed for x in reasons[left+1:right,j]):continue
            for i in range(left+1,right):
                fraction=(times[i]-times[left])/(times[right]-times[left])
                corrected[i,j]=corrected[left,j]*(1-fraction)+corrected[right,j]*fraction
                states[i,j]='interpolated'
    # Causal exponential low-pass, reset at every missing joint/target gap.
    if settings.smoothing_tau>0:
        for j in range(23):
            prev=None
            influence=0.0
            for i in range(n):
                if not np.isfinite(corrected[i,j]).all():
                    prev=None;influence=0.0;continue
                is_interpolated=states[i,j]=='interpolated'
                if prev is not None and times[i]-times[prev]<=settings.max_derivative_gap:
                    alpha=1-np.exp(-(times[i]-times[prev])/settings.smoothing_tau)
                    corrected[i,j]=alpha*corrected[i,j]+(1-alpha)*corrected[prev,j]
                    influence=(1-alpha)*influence
                    if states[i,j]=='observed':states[i,j]='smoothed'
                    if influence>0.01 and not is_interpolated:
                        states[i,j]='smoothed_after_interpolation'
                else:influence=0.0
                if is_interpolated:influence=1.0
                prev=i
    return raw,corrected,reasons,states

def metrics(frames,raw,corrected,states,settings):
    rows=[]
    def provenance(state):
        if 'interpolated' in state or 'smoothed_after_interpolation' in state:return 'interpolated_influence'
        if 'smoothed' in state:return 'smoothed'
        return 'observed'
    for i,r in enumerate(frames):
        row={'frame_index':r['index'],'time_seconds':r['time'],'track_state':r['track_state']}
        for name,indices in ANGLES.items():
            row[name+'_raw_deg']=angle3(*raw[i,list(indices)]) if all(r['reasons'][j]=='valid' for j in indices) else np.nan
            row[name+'_corrected_deg']=angle3(*corrected[i,list(indices)])
            row[name+'_state']=provenance(states[i,list(indices)]) if np.isfinite(row[name+'_corrected_deg']) else 'unavailable'
        for label,arr in [('raw',raw),('corrected',corrected)]:
            points=arr[i,[5,6,11,12]]
            if label=='raw' and not all(r['reasons'][j]=='valid' for j in [5,6,11,12]):points=np.full((4,2),np.nan)
            if np.isfinite(points).all():
                shoulders=points[:2].mean(axis=0);hips=points[2:].mean(axis=0)
                v=shoulders-hips
                tilt=np.degrees(np.arctan2(v[0],-v[1]))
                row['trunk_'+label+'_deg']=(tilt-settings.camera_roll_degrees+180)%360-180
            else:row['trunk_'+label+'_deg']=np.nan
        row['trunk_state']=provenance(states[i,[5,6,11,12]]) if np.isfinite(row['trunk_corrected_deg']) else 'unavailable'
        rows.append(row)
    table=pd.DataFrame(rows)
    # Pairwise angular derivative, never cross a target or long sampling gap.
    for side in ('left','right'):
        name=side+'_knee'
        for variant in ('raw','corrected'):
            velocity=np.full(len(table),np.nan)
            for i in range(1,len(table)):
                dt=table.time_seconds[i]-table.time_seconds[i-1]
                values=table[name+'_'+variant+'_deg'].iloc[i-1:i+1].to_numpy()
                if (0<dt<=settings.max_derivative_gap and np.isfinite(values).all()
                    and table.track_state[i]=='locked' and table.track_state[i-1]=='locked'):
                    velocity[i]=(values[1]-values[0])/dt
            table[name+'_'+variant+'_deg_per_sec']=velocity
        table[name+'_velocity_state']=[
            'unavailable' if not np.isfinite(table[name+'_corrected_deg_per_sec'][i]) else
            ('interpolated_influence' if any(table[name+'_state'][j]=='interpolated_influence' for j in [i-1,i]) else 'derived')
            for i in range(len(table))]
    return table

def coordinate_table(frames,raw,corrected,reasons,states):
    rows=[]
    for i,r in enumerate(frames):
        p=r['pose']
        for j,name in enumerate(NAMES):
            rows.append({'frame_index':r['index'],'time_seconds':r['time'],'joint':name,
                         'raw_x':raw[i,j,0],'raw_y':raw[i,j,1],
                         'corrected_x':corrected[i,j,0],'corrected_y':corrected[i,j,1],
                         'score':p.score[j],'score_kind':p.score_kind,
                         'presence':p.presence[j],'visibility':p.visibility[j],
                         'rejection_reason':reasons[i,j],'corrected_state':states[i,j],
                         'track_state':r['track_state'], 'association_cost':r['association_cost']})
    return pd.DataFrame(rows)
