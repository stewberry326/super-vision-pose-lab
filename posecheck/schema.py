from dataclasses import dataclass, asdict
import numpy as np

BODY_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
              'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
FOOT_NAMES = ['left_big_toe', 'left_small_toe', 'left_heel',
              'right_big_toe', 'right_small_toe', 'right_heel']
NAMES = BODY_NAMES + FOOT_NAMES
LABELS = ['코', '왼눈', '오른눈', '왼귀', '오른귀', '왼어깨', '오른어깨',
          '왼팔꿈치', '오른팔꿈치', '왼손목', '오른손목', '왼골반', '오른골반',
          '왼무릎', '오른무릎', '왼발목', '오른발목', '왼엄지발끝', '왼새끼발끝',
          '왼뒤꿈치', '오른엄지발끝', '오른새끼발끝', '오른뒤꿈치']
EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
         (11,13),(13,15),(12,14),(14,16),(15,17),(15,19),(17,18),
         (16,20),(16,22),(20,21)]
ANGLES = {'left_elbow': (5,7,9), 'right_elbow': (6,8,10),
          'left_knee': (11,13,15), 'right_knee': (12,14,16)}

@dataclass
class Settings:
    start: float = 0.0
    end: float = 10.0
    sample_fps: float = 10.0
    confidence: float = 0.35
    presence: float = 0.50
    visibility: float = 0.50
    max_gap_seconds: float = 0.25
    max_speed_body_per_second: float = 5.0
    bone_change_ratio: float = 0.65
    smoothing_tau: float = 0.06
    max_derivative_gap: float = 0.25
    max_track_gap: float = 0.50
    camera_roll_degrees: float = 0.0
    condition: str = '근접·부분 신체'
    max_output_width: int = 1280
    flip_test: bool = True

    def to_dict(self):
        return asdict(self)

@dataclass
class Pose:
    xy: np.ndarray
    score: np.ndarray
    presence: np.ndarray
    visibility: np.ndarray
    score_kind: str

    @classmethod
    def empty(cls):
        return cls(np.full((23,2), np.nan), np.full(23,np.nan),
                   np.full(23,np.nan), np.full(23,np.nan), 'unavailable')
