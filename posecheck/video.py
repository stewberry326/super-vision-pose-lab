from pathlib import Path
import json
import subprocess
import cv2
import numpy as np

def probe(path):
    path = str(Path(path).expanduser().resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    data = json.loads(subprocess.check_output([
        'ffprobe','-v','error','-select_streams','v:0','-show_packets',
        '-show_entries','packet=pts_time:stream=width,height,avg_frame_rate:format=duration',
        '-of','json',path], text=True))
    packets = data.get('packets', [])
    timestamps = np.sort(np.array([float(x['pts_time']) for x in packets if 'pts_time' in x], dtype=float))
    cap=cv2.VideoCapture(path)
    count=round(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    timestamp_source='sorted packet PTS (one packet per video frame, frame count checked)'
    if count!=len(timestamps):
        # Nonstandard containers/codecs can have multiple frames per packet.
        frame_data=json.loads(subprocess.check_output([
            'ffprobe','-v','error','-select_streams','v:0','-show_frames',
            '-show_entries','frame=best_effort_timestamp_time','-of','json',path],text=True))
        timestamps=np.array([float(x['best_effort_timestamp_time']) for x in frame_data['frames']])
        timestamp_source='decoded frame best_effort_timestamp_time'
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError('유효한 프레임 타임스탬프를 읽지 못했습니다. ffmpeg로 영상을 다시 저장해주세요.')
    first_pts = float(timestamps[0])
    timestamps -= first_pts
    stream = data['streams'][0]
    return {'path':path, 'width':stream['width'], 'height':stream['height'],
            'duration':float(data['format']['duration']), 'timestamps':timestamps,
            'source_first_pts':first_pts, 'frames':len(timestamps),'timestamp_source':timestamp_source}

def frame_at(meta, seconds):
    index = int(np.argmin(np.abs(meta['timestamps'] - seconds)))
    cap = cv2.VideoCapture(meta['path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f'{seconds:.2f}초 프레임을 읽지 못했습니다.')
    return index, float(meta['timestamps'][index]), frame

def sample_indices(meta, start, end, fps):
    t = meta['timestamps']
    valid = np.flatnonzero((t >= start) & (t <= end))
    selected = []
    next_t = start
    for index in valid:
        if t[index] + 1e-6 >= next_t:
            selected.append(int(index))
            next_t = t[index] + 1.0 / fps - 1e-6
    return selected

def iter_frames(meta, indices):
    cap = cv2.VideoCapture(meta['path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    wanted = set(indices)
    for i in range(indices[0], indices[-1] + 1):
        if i in wanted:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise ValueError(f'프레임 {i} 디코딩 실패')
            yield i, float(meta['timestamps'][i]), frame
        else:
            if not cap.grab():
                cap.release()
                raise ValueError(f'프레임 {i} 디코딩 실패')
    cap.release()
