import numpy as np
import pytest
from posecheck.schema import Pose,Settings
from posecheck.processing import angle3,classify,correct,metrics
from posecheck.tracking import TargetLock
from posecheck.video import sample_indices

def rows(times,missing=None,tracking=None):
    result=[]
    for i,t in enumerate(times):
        p=Pose.empty();p.xy[:17]=np.array([50+t*10,50]);p.score[:17]=0.9
        reasons=['valid']*17+['unsupported_or_missing']*6
        if missing and i in missing:
            reasons[7]=missing[i];p.xy[7]=[500,500]
        state=tracking.get(i,'locked') if tracking else 'locked'
        if state!='locked':reasons=['track_lost']*23
        result.append({'index':i,'time':t,'pose':p,'reasons':reasons,
                       'track_state':state,'box':np.array([0,0,100,200]),'association_cost':0.1})
    return result

def config(**kwargs):return Settings(smoothing_tau=0,bone_change_ratio=10,**kwargs)

def test_angle_geometry_and_degenerate():
    assert angle3([0,0],[1,0],[2,0])==pytest.approx(180)
    assert angle3([0,1],[0,0],[1,0])==pytest.approx(90)
    assert np.isnan(angle3([0,0],[0,0],[1,0]))

def test_short_gap_uses_real_time():
    r=rows([0,0.04,0.18],{1:'low_score'})
    raw,clean,reason,state=correct(r,config(max_gap_seconds=0.2))
    assert raw[1,7,0]==500
    assert clean[1,7,0]==pytest.approx(50.4)
    assert state[1,7]=='interpolated'
    assert reason[1,7]=='low_score'

@pytest.mark.parametrize('reason',['out_of_frame','low_presence','unsupported_or_missing'])
def test_forbidden_gap_not_filled(reason):
    _,clean,_,state=correct(rows([0,.1,.2],{1:reason}),config())
    assert np.isnan(clean[1,7]).all()
    assert state[1,7]=='unavailable'

def test_long_gap_not_filled():
    _,clean,_,_=correct(rows([0,.1,.8],{1:'low_score'}),config())
    assert np.isnan(clean[1,7]).all()

def test_target_loss_is_interpolation_boundary():
    _,clean,_,_=correct(rows([0,.1,.2],tracking={1:'lost'}),config())
    assert np.isnan(clean[1]).all()

def test_spike_does_not_destroy_following_point():
    r=rows([0,.1,.2],{1:'valid'})
    _,clean,reason,state=correct(r,config(max_speed_body_per_second=2))
    assert reason[1,7]=='outlier_speed'
    assert reason[2,7]=='valid'
    assert clean[1,7,0]==pytest.approx(51)

def test_outside_and_separate_presence_visibility():
    p=Pose.empty();p.xy[:17]=[50,50];p.score[:17]=0.9
    p.xy[5]=[-1,50];p.presence[6]=.1;p.visibility[7]=.1
    status=classify(p,None,(100,100,3),Settings(),'locked')
    assert status[5]=='out_of_frame'
    assert status[6]=='low_presence'
    assert status[7]=='low_visibility'
    assert status[8]=='valid' # RTMW unknown visibility must remain unknown, not manufactured.

def test_derivative_uses_timestamps_and_stops_at_loss():
    r=rows([0,.1,.25,.35],tracking={3:'lost'})
    for i,x in enumerate(r):
        degree=90+i*10
        x['pose'].xy[[11,13,15]]=[[1,0],[0,0],[np.cos(np.radians(degree)),np.sin(np.radians(degree))]]
    raw,clean,_,state=correct(r,config(max_speed_body_per_second=50))
    table=metrics(r,raw,clean,state,config())
    assert table.left_knee_raw_deg_per_sec[1]==pytest.approx(100)
    assert table.left_knee_raw_deg_per_sec[2]==pytest.approx(10/.15)
    assert np.isnan(table.left_knee_raw_deg_per_sec[3])

def test_track_does_not_change_to_far_person():
    f=np.full((200,300,3),100,np.uint8)
    lock=TargetLock([10,10,90,180],max_gap=.3)
    b,s,_=lock.update(f,[[10,10,90,180]],0)
    assert s=='locked'
    b,s,_=lock.update(f,[[200,10,290,180]],.1)
    assert b is None and s=='lost'
    b,s,_=lock.update(f,[[10,10,90,180]],.5)
    assert b is None and s=='reselect_required'

def test_ambiguous_people_are_not_auto_selected():
    f=np.full((200,300,3),100,np.uint8)
    lock=TargetLock([10,10,90,180])
    _,s,_=lock.update(f,[[10,10,90,180],[15,10,95,180]],0)
    assert s=='ambiguous'

def test_vfr_sampling_no_synthetic_timestamps():
    meta={'timestamps':np.array([0,.03,.11,.18,.24,.40])}
    assert sample_indices(meta,0,.4,10)==[0,2,4,5]
