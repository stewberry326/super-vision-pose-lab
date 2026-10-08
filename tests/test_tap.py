import json
import pytest
import numpy as np
import pandas as pd
from posecheck.tap import choose_joint
from posecheck.annotations import compare

def test_strong_disagreement_is_not_averaged_or_overwritten():
    p,s,r,reset=choose_joint(np.array([100.,100.]),True,np.array([200.,200.]),True,.2,.06)
    assert np.isnan(p).all() and s=='unavailable' and not reset

def test_reliable_detection_restarts_failed_tracker():
    p,s,r,reset=choose_joint(np.array([100.,100.]),True,np.array([200.,200.]),False,.2,.06)
    assert np.all(p==[100,100]) and reset

def test_tracker_survives_bad_detection():
    p,s,r,reset=choose_joint(np.array([100.,100.]),False,np.array([200.,200.]),True,.2,.06)
    assert np.all(p==[200,200]) and not reset and s=='tracked'

def test_agreement_keeps_track():
    p,s,r,reset=choose_joint(np.array([100.,100.]),True,np.array([102.,100.]),True,.01,.06)
    assert np.all(p==[102,100]) and not reset

def test_heldout_excludes_manual_inputs_and_reports_coverage(tmp_path):
    (tmp_path/'ProbPose-TAP').mkdir()
    (tmp_path/'report.json').write_text(json.dumps({'models':{'ProbPose-TAP':{}},'source':{'width':1000,'height':1000},'settings':{}}))
    pd.DataFrame([
        {'frame_index':0,'joint':'left_knee','x':10,'y':10,'visible':True,'split':'evaluation'},
        {'frame_index':1,'joint':'left_knee','x':10,'y':10,'visible':True,'split':'evaluation'},
        {'frame_index':2,'joint':'left_knee','x':10,'y':10,'visible':True,'split':'evaluation'},
        {'frame_index':3,'joint':'left_knee','x':10,'y':10,'visible':True,'split':'calibration'},
        {'frame_index':4,'joint':'left_knee','x':np.nan,'y':np.nan,'visible':False,'split':'evaluation'}
    ]).to_csv(tmp_path/'annotations.csv',index=False)
    (tmp_path/'tap_seeds.json').write_text(json.dumps([{'frame_index':0,'joint':'left_knee','x':10,'y':10}]))
    rows=[]
    for i in range(5):
        x=np.nan if i==2 else 13
        rows.append({'frame_index':i,'joint':'left_knee','raw_x':x,'raw_y':14,'corrected_x':x,'corrected_y':14,'rejection_reason':'valid' if i!=2 else 'track_unreliable'})
    pd.DataFrame(rows).to_csv(tmp_path/'ProbPose-TAP/coordinates.csv',index=False)
    pd.DataFrame({'frame_index':range(5)}).to_csv(tmp_path/'ProbPose-TAP/metrics.csv',index=False)
    summary,details=compare(tmp_path)
    allrow=summary[summary.condition=='all'].iloc[0]
    assert allrow.visible_references==2 and allrow.accepted_predictions==1
    assert allrow.coverage==.5 and allrow.mean_error_px==5
    assert allrow.unseen_but_accepted_rate==1
    assert set(details.frame_index)=={1,2,4}

def test_delay_requires_dense_moving_labels():
    from posecheck.annotations import estimate_lag
    assert estimate_lag([0,1],[[0,0],[1,0]],[0,1],[[0,0],[1,0]])['status']=='insufficient_dense_references'
    t=np.arange(0,3,.1);r=np.stack([100*t,10*np.sin(t)],-1)
    pred_t=np.arange(0,3.5,.05);pred=np.stack([100*(pred_t-.1),10*np.sin(pred_t-.1)],-1)
    result=estimate_lag(t,r,pred_t,pred)
    assert result['lag_seconds']==pytest.approx(.1,abs=.011)


def test_known_conflict_cannot_be_revived_by_tracker_confidence_alone():
    p,s,r,restart=choose_joint(np.array([100.,100.]),False,np.array([200.,200.]),True,.2,.06,conflicted=True)
    assert np.isnan(p).all() and s=='unavailable'
    p,s,r,restart=choose_joint(np.array([100.,100.]),True,np.array([200.,200.]),True,.2,.06,conflicted=True)
    assert restart and s=='redetected' and np.all(p==[100,100])
