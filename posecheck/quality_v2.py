"""Experimental causal quality gate; image geometry is evidence, not anatomy.

Original inference is immutable. Segment warnings never select an endpoint by
edge order. Only a segment anomaly plus a clearly isolated endpoint motion can
reject that endpoint. Uncertain points remain visible but not measurable.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
import pandas as pd
from .schema import EDGES, NAMES, ANGLES
from .processing import correct, metrics


@dataclass
class QualitySettings:
    history_seconds: float = 1.0
    minimum_history: int = 5
    motion_floor_body: float = .045
    motion_mad_factor: float = 6.0
    minimum_motion_body_per_second: float = .5
    endpoint_dominance: float = 2.5
    comparison_gap_seconds: float = .4
    minimum_segment_body: float = .03

    def to_dict(self):
        return asdict(self)


def correct_v2(frames, settings, quality=None):
    quality = quality or QualitySettings()
    n = len(frames)
    times = np.array([r['time'] for r in frames])
    raw = np.array([r['pose'].xy for r in frames])
    base = np.array([r['reasons'] for r in frames], dtype=object)
    reasons = base.copy()
    scale = np.array([max(np.linalg.norm(r['box'][2:]-r['box'][:2]),40)
                      if r['box'] is not None else np.nan for r in frames])
    valid = base == 'valid'
    motion = np.full((n,23),np.nan)
    thresholds = np.full((n,23),np.nan)
    motion_flag = np.zeros((n,23),bool)
    edge_flag = np.zeros((n,len(EDGES)),bool)
    edge_rows = []
    cumulative = np.zeros(2)
    common = np.zeros((n,2))
    last = np.full(23,-1,int)
    last_cumulative = np.zeros((23,2))
    movement_history = [[] for _ in NAMES]
    length_history = [[] for _ in EDGES]
    for i,r in enumerate(frames):
        if r['track_state'] != 'locked' or not np.isfinite(scale[i]):
            last[:] = -1; cumulative[:] = 0; last_cumulative[:] = 0
            movement_history = [[] for _ in NAMES]
            length_history = [[] for _ in EDGES]
            continue
        # Median translation is only an image-motion reference. It does not
        # separate patient movement from head/camera movement or infer 3D.
        if i and times[i]-times[i-1] <= quality.comparison_gap_seconds:
            shared = valid[i,5:17] & valid[i-1,5:17]
            if shared.sum() >= 4:
                common[i] = np.median(raw[i,5:17][shared]-raw[i-1,5:17][shared],axis=0)
        cumulative += common[i]
        for j in range(23):
            if not valid[i,j]:
                continue
            movement_history[j] = [(t,v) for t,v in movement_history[j]
                                   if times[i]-t <= quality.history_seconds]
            prev = last[j]
            if prev >= 0 and 0 < times[i]-times[prev] <= quality.comparison_gap_seconds:
                dt = times[i]-times[prev]
                expected = raw[prev,j] + cumulative-last_cumulative[j]
                motion[i,j] = np.linalg.norm(raw[i,j]-expected)/max(scale[i],scale[prev])
                speeds = np.array([v for _,v in movement_history[j]])
                adaptive = 0.
                if len(speeds) >= quality.minimum_history:
                    med = np.median(speeds)
                    mad = 1.4826*np.median(abs(speeds-med))
                    adaptive = (med + quality.motion_mad_factor*mad)*dt
                thresholds[i,j] = max(quality.motion_floor_body,
                                     quality.minimum_motion_body_per_second*dt,adaptive)
                motion_flag[i,j] = motion[i,j] > thresholds[i,j]
        # Histories use recent base-valid RAW pairs, including warned lengths.
        # This avoids freezing an early pose as the forever reference.
        rejected = np.zeros(23,bool)
        for e,(a,b) in enumerate(EDGES):
            history = [(t,v) for t,v in length_history[e]
                       if times[i]-t <= quality.history_seconds]
            length_history[e] = history
            if not (valid[i,a] and valid[i,b]):
                continue
            length = np.linalg.norm(raw[i,a]-raw[i,b])/scale[i]
            median = np.median([v for _,v in history]) if history else np.nan
            change = length/median-1 if median>quality.minimum_segment_body else np.nan
            anomaly = len(history)>=quality.minimum_history and np.isfinite(change) and abs(change)>settings.bone_change_ratio
            edge_flag[i,e] = anomaly
            culprit = None
            if anomaly:
                for suspect,partner in [(a,b),(b,a)]:
                    if (motion_flag[i,suspect] and np.isfinite(motion[i,partner])
                        and not motion_flag[i,partner]
                        and motion[i,suspect] > quality.endpoint_dominance*max(motion[i,partner],.01)):
                        culprit = suspect; rejected[suspect] = True
                edge_rows.append({'frame_index':r['index'],'time_seconds':times[i],
                    'edge_index':e,'joint_a':NAMES[a],'joint_b':NAMES[b],
                    'current_length_body':length,'recent_median_body':median,
                    'relative_change':change,'history_pairs':len(history),
                    'suspect_endpoint':NAMES[culprit] if culprit is not None else '',
                    'decision':'endpoint_rejected' if culprit is not None else 'segment_warning_only'})
            length_history[e].append((times[i],length))
        for j in range(23):
            if not valid[i,j]:
                continue
            if rejected[j]:
                reasons[i,j] = 'outlier_isolated_motion_and_segment'
            elif motion_flag[i,j]:
                reasons[i,j] = 'uncertain_motion'
            else:
                prev = last[j]
                if prev>=0 and 0<times[i]-times[prev]<=quality.comparison_gap_seconds and np.isfinite(motion[i,j]):
                    movement_history[j].append((times[i],motion[i,j]/(times[i]-times[prev])))
                last[j]=i; last_cumulative[j]=cumulative.copy()
    # Reuse bounded gap/smoothing mechanics, disable both legacy geometry gates.
    rows = [{**r,'reasons':list(reasons[i])} for i,r in enumerate(frames)]
    neutral = replace(settings,bone_change_ratio=float('inf'),max_speed_body_per_second=float('inf'))
    _,clean,reasons,states = correct(rows,neutral)
    # No interpolation of new structural/motion warnings. Measurements exclude
    # those points; displayed uncertain points are their untouched RAW positions.
    display = clean.copy(); display_states = states.copy()
    for i in range(n):
        for j in range(23):
            if (reasons[i,j] in {'uncertain_motion','low_score','low_visibility'}
                and np.isfinite(raw[i,j]).all() and not np.isfinite(clean[i,j]).all()):
                display[i,j] = raw[i,j]
                display_states[i,j] = 'uncertain'
    table = metrics(frames,raw,clean,states,settings)
    edge_lookup = {frozenset(pair):e for e,pair in enumerate(EDGES)}
    for name,(a,b,c) in ANGLES.items():
        warn = edge_flag[:,edge_lookup[frozenset((a,b))]] | edge_flag[:,edge_lookup[frozenset((b,c))]]
        table[name+'_display_deg'] = table[name+'_corrected_deg']
        table[name+'_geometry_warning'] = warn
        table.loc[warn,name+'_corrected_deg'] = np.nan
        table.loc[warn,name+'_state'] = 'geometry_warning'
    torso = [edge_lookup[frozenset(pair)] for pair in [(5,6),(5,11),(6,12),(11,12)]]
    torso_warn = edge_flag[:,torso].any(axis=1)
    table['trunk_display_deg'] = table.trunk_corrected_deg
    table['trunk_geometry_warning'] = torso_warn
    table.loc[torso_warn,'trunk_corrected_deg'] = np.nan
    table.loc[torso_warn,'trunk_state'] = 'geometry_warning'
    # Derivatives must not silently retain the pre-warning angle calculation.
    for side in ['left','right']:
        value = table[side+'_knee_corrected_deg'].to_numpy()
        for i in range(n):
            if i==0 or not np.isfinite(value[max(0,i-1):i+1]).all():
                table.loc[i,side+'_knee_corrected_deg_per_sec']=np.nan
                table.loc[i,side+'_knee_velocity_state']='unavailable'
    diagnostics=[]
    for i,r in enumerate(frames):
        for j,name in enumerate(NAMES):
            linked=[e for e,pair in enumerate(EDGES) if j in pair]
            diagnostics.append({'frame_index':r['index'],'time_seconds':times[i],'joint':name,
                'base_reason':base[i,j],'motion_body':motion[i,j],
                'motion_threshold_body':thresholds[i,j],
                'common_translation_x':common[i,0],'common_translation_y':common[i,1],
                'segment_warning_count':int(edge_flag[i,linked].sum()),
                'v2_reason':reasons[i,j],'display_state':display_states[i,j],
                'measurement_available':bool(np.isfinite(clean[i,j]).all())})
    return {'raw':raw,'corrected':clean,'reasons':reasons,'states':states,
        'display':display,'display_states':display_states,'edge_warnings':edge_flag,
        'metrics':table,'diagnostics':pd.DataFrame(diagnostics),
        'edge_diagnostics':pd.DataFrame(edge_rows,columns=['frame_index','time_seconds','edge_index','joint_a','joint_b','current_length_body','recent_median_body','relative_change','history_pairs','suspect_endpoint','decision']),
        'settings':quality.to_dict()}
