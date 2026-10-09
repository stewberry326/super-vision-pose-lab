import numpy as np
from posecheck.schema import Settings
from posecheck.quality_v2 import correct_v2
from posecheck.quality_v3 import correct_v3
from test_quality_v2 import scene


def test_coherent_limb_translation_survives_without_inventing_coordinates():
    rows=scene(16)
    for i in range(7,16):rows[i]['pose'].xy[[7,9]]+=[60,0]
    s=Settings(smoothing_tau=0)
    old=correct_v2(rows,s);new=correct_v3(rows,s)
    assert old['reasons'][7,9]=='uncertain_motion'
    assert new['reasons'][7,9]=='valid'
    assert new['diagnostics'].query("frame_index==7 and joint=='left_wrist'").coherent_chain_support.item()
    assert np.array_equal(new['corrected'][7,9],rows[7]['pose'].xy[9])


def test_isolated_spike_rejected_and_true_return_recovers_immediately():
    rows=scene();rows[7]['pose'].xy[9]=[500,200]
    out=correct_v3(rows,Settings(smoothing_tau=0))
    assert out['reasons'][7,9]=='outlier_isolated_motion_and_segment'
    assert np.isnan(out['display'][7,9]).all()
    assert out['reasons'][8,9]=='valid'


def test_consistent_plausible_candidate_requires_three_observations():
    rows=scene(20)
    for i in range(7,20):rows[i]['pose'].xy[9]=[160,190]
    out=correct_v3(rows,Settings(smoothing_tau=0))
    assert (out['reasons'][7:9,9]=='uncertain_motion').all()
    assert out['reasons'][9,9]=='valid'
    assert out['diagnostics'].query("frame_index==9 and joint=='left_wrist'").reacquired.item()


def test_repeated_impossible_spike_does_not_pass_when_clock_expires():
    rows=scene(24)
    for i in range(7,24):rows[i]['pose'].xy[9]=[500,200]
    out=correct_v3(rows,Settings(smoothing_tau=0))
    assert np.isnan(out['corrected'][7:,9]).all()
    assert not out['diagnostics'].query("joint=='left_wrist'").reacquired.any()


def test_constant_motion_uses_directional_prediction():
    rows=scene(16)
    for i in range(7,16):rows[i]['pose'].xy[9]+=[(i-6)*22,0]
    out=correct_v3(rows,Settings(smoothing_tau=0))
    assert out['diagnostics'].query("frame_index==9 and joint=='left_wrist'").prediction_used.item()
    assert out['reasons'][7:10,9].tolist()==['valid']*3


def test_target_loss_resets_velocity_and_pending_candidate():
    rows=scene(16)
    rows[7]['pose'].xy[9]=[160,190]
    rows[8]['track_state']='lost';rows[8]['reasons']=['track_lost']*23
    for i in range(9,16):rows[i]['pose'].xy[9]=[120,200]
    out=correct_v3(rows,Settings(smoothing_tau=0))
    assert np.isnan(out['display'][8]).all()
    row=out['diagnostics'].query("frame_index==9 and joint=='left_wrist'")
    assert not row.prediction_used.item() and not row.reacquired.item()


def test_outside_and_low_presence_are_never_restored_by_coherence():
    rows=scene()
    rows[7]['reasons'][9]='out_of_frame';rows[7]['pose'].xy[9]=[-5,200]
    rows[8]['reasons'][9]='low_presence'
    out=correct_v3(rows,Settings(smoothing_tau=0))
    assert np.isnan(out['display'][7:9,9]).all()


def test_prefix_invariance_of_causal_gate_and_recovery():
    rows=scene(20)
    for i in range(7,20):rows[i]['pose'].xy[9]=[160,190]
    s=Settings(smoothing_tau=0,max_gap_seconds=0)
    full=correct_v3(rows,s)
    for end in [8,9,10,14]:
        prefix=correct_v3(rows[:end],s)
        np.testing.assert_array_equal(prefix['reasons'],full['reasons'][:end])
        np.testing.assert_allclose(prefix['corrected'],full['corrected'][:end],equal_nan=True)


def test_measurement_excludes_all_uncertain_and_warned_angles():
    rows=scene();rows[7]['pose'].xy[9]=[160,190]
    out=correct_v3(rows,Settings(smoothing_tau=0))
    mask=out['display_states']=='uncertain'
    assert np.isnan(out['corrected'][mask]).all()
    for name in ['left_elbow','right_elbow','left_knee','right_knee']:
        warn=out['metrics'][name+'_geometry_warning']
        assert out['metrics'].loc[warn,name+'_corrected_deg'].isna().all()


def test_real_missing_gap_requires_fresh_confirmation_not_stale_segment():
    rows=scene(30)
    for i in range(7,15):
        rows[i]['reasons'][9]='out_of_frame'
        rows[i]['pose'].xy[9]=[-10,200]
    for i in range(15,30):rows[i]['pose'].xy[9]=[160,190]
    out=correct_v3(rows,Settings(smoothing_tau=0))
    assert np.isnan(out['corrected'][7:17,9]).all()
    assert out['reasons'][17,9]=='valid'
    assert out['diagnostics'].query("frame_index==17 and joint=='left_wrist'").reacquired.item()


def test_common_pan_does_not_reject_body():
    rows=scene()
    for i in range(7,len(rows)):rows[i]['pose'].xy[5:17]+=[200,100]
    out=correct_v3(rows,Settings(smoothing_tau=0))
    assert (out['reasons'][7,5:17]=='valid').all()


def test_reprocess_v3_preserves_v2_and_writes_metric_comparison(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from posecheck import reprocess as module
    from posecheck.schema import Pose
    rows=scene();s=Settings(smoothing_tau=0)
    v2=correct_v2(rows,s)
    old={**v2,'indices':np.arange(len(rows)),'times':np.array([r['time'] for r in rows]),'boxes':np.array([r['box'] for r in rows])}
    base=tmp_path/'baseline';(base/'ProbPose-s').mkdir(parents=True)
    (base/'ProbPose-s/raw.mp4').write_bytes(b'test')
    (base/'ProbPose-v2').mkdir()
    np.savez(base/'ProbPose-v2/arrays.npz',**{k:(v2[k].astype(str) if k=='display_states' else v2[k]) for k in ['corrected','display','display_states','edge_warnings']})
    summary={'raw_valid_rates':{'left_knee':1.,'right_knee':1.}}
    report={'settings':s.to_dict(),'source':{'path':'test.mp4','duration':2},'models':{'ProbPose-s':summary,'ProbPose-v2':summary}}
    monkeypatch.setattr(module,'restore_rows',lambda _: (report,s,old,None,rows))
    monkeypatch.setattr(module,'probe',lambda _:SimpleNamespace())
    monkeypatch.setattr(module,'iter_frames',lambda *_: ((i,r['time'],np.zeros((36,64,3),np.uint8)) for i,r in enumerate(rows)))
    monkeypatch.setattr(module,'encode_frames',lambda source,times,path,end:path.write_bytes(b'test'))
    monkeypatch.setattr(module,'graph_png',lambda table,path:path.write_bytes(b'test'))
    out=module.reprocess(base,version=3)
    loaded=json.loads((out/'report.json').read_text())
    assert loaded['quality_experiment']['version']==3
    assert loaded['quality_experiment']['comparison_layout'].startswith('2x2')
    assert list(loaded['models'])==['ProbPose-s','ProbPose-v2','ProbPose-v3']
    assert (out/'ProbPose-v2/arrays.npz').read_bytes()==(base/'ProbPose-v2/arrays.npz').read_bytes()
    import pandas as pd
    table=pd.read_csv(out/'ProbPose-v3/metrics.csv')
    assert 'left_knee_raw_deg' in table
    assert 'v3_reason' in pd.read_csv(out/'ProbPose-v3/coordinates.csv')
    assert 'v2_measurement_points' in pd.read_csv(out/'rule_comparison.csv')
