import numpy as np
from posecheck.schema import Pose,Settings
from posecheck.quality_v2 import correct_v2


def scene(n=12,dt=.1):
    points=np.full((23,2),np.nan)
    points[5:17]=[[100,100],[180,100],[100,150],[180,150],
                  [100,200],[180,200],[100,250],[180,250],
                  [100,320],[180,320],[100,390],[180,390]]
    rows=[]
    for i in range(n):
        p=Pose.empty();p.xy=points.copy();p.score[5:17]=.9
        rows.append({'index':i,'time':i*dt,'pose':p,
            'box':np.array([0.,0.,300.,500.]),'track_state':'locked',
            'association_cost':.1,'reasons':['unsupported_or_missing']*5+['valid']*12+['unsupported_or_missing']*6})
    return rows


def test_length_change_alone_warns_segment_not_endpoint():
    rows=scene()
    # An abrupt foreshortening-like change is kept as uncertain motion/edge,
    # never rejected solely because the edge's second endpoint was named last.
    for i,y in [(7,135),(8,120),(9,105)]:rows[i]['pose'].xy[7]=[100,y]
    r=correct_v2(rows,Settings(smoothing_tau=0))
    assert np.isfinite(r['display'][9,7]).all()
    assert r['reasons'][9,7]!='outlier_isolated_motion_and_segment'
    assert r['edge_warnings'][9].any()
    assert np.isnan(r['metrics'].left_elbow_corrected_deg.iloc[9])


def test_isolated_spike_rejects_culprit_not_stable_partner_and_recovers():
    rows=scene();rows[7]['pose'].xy[9]=[500,200]
    r=correct_v2(rows,Settings(smoothing_tau=0))
    assert r['reasons'][7,9]=='outlier_isolated_motion_and_segment'
    assert np.isnan(r['display'][7,9]).all()
    assert np.isfinite(r['corrected'][7,7]).all()
    assert np.isfinite(r['corrected'][8,9]).all()


def test_large_common_translation_does_not_reject_every_joint():
    rows=scene()
    for i in range(7,len(rows)):rows[i]['pose'].xy[5:17]+=[200,100]
    r=correct_v2(rows,Settings(smoothing_tau=0))
    assert (r['reasons'][7,5:17]=='valid').all()
    assert np.isfinite(r['corrected'][7,5:17]).all()


def test_recent_history_adapts_instead_of_freezing_first_pose():
    rows=scene(30)
    for i in range(7,len(rows)):rows[i]['pose'].xy[7]=[100,115]
    r=correct_v2(rows,Settings(smoothing_tau=0))
    assert not r['edge_warnings'][-1].any()
    assert r['reasons'][-1,7]=='valid'


def test_hard_outside_boundary_not_restored_for_display():
    rows=scene();rows[7]['reasons'][9]='out_of_frame';rows[7]['pose'].xy[9]=[-10,200]
    r=correct_v2(rows,Settings(smoothing_tau=0))
    assert np.isnan(r['display'][7,9]).all() and np.isnan(r['corrected'][7,9]).all()


def test_motion_uncertainty_is_visible_but_not_measured():
    rows=scene();rows[7]['pose'].xy[9]=[160,190]
    r=correct_v2(rows,Settings(smoothing_tau=0))
    assert r['reasons'][7,9]=='uncertain_motion'
    assert np.isfinite(r['display'][7,9]).all() and np.isnan(r['corrected'][7,9]).all()
    assert r['display_states'][7,9]=='uncertain'


def test_review_queue_covers_interval_and_keeps_known_cases_out_of_evaluation():
    from posecheck.reprocess import make_review
    rows=scene(200)
    for r in rows:r['time']+=14.
    v=correct_v2(rows,Settings(smoothing_tau=0))
    old={'corrected':v['corrected'],'reasons':v['reasons']}
    frames,queue=make_review(old,v,rows,14.)
    assert len(frames)==200 and not frames.candidate.any()
    assert queue.time_seconds.min()==14. and queue.time_seconds.max()>=33.
    assert queue.selection_source.str.contains('fixed_random_sample').any()
    assert queue.loc[queue.time_seconds.between(17.5,21.99),'suggested_split'].eq('calibration').all()
    assert queue.loc[queue.time_seconds.eq(20.5),'selection_source'].str.contains('user_reported_example').all()
