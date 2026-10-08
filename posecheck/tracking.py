"""Conservative detector association, not guaranteed biometric identification."""
import cv2
import numpy as np

def iou(a,b):
    p = np.maximum(a[:2], b[:2]); q = np.minimum(a[2:],b[2:])
    intersection = np.prod(np.maximum(0,q-p))
    area = np.prod(np.maximum(0,a[2:]-a[:2])) + np.prod(np.maximum(0,b[2:]-b[:2]))
    return float(intersection / max(area-intersection,1))

def appearance(frame, box):
    h,w = frame.shape[:2]
    x0,y0,x1,y1 = np.asarray(box,dtype=int)
    # Central clothing region; exclude much of the background.
    dx = int((x1-x0)*0.18); dy = int((y1-y0)*0.18)
    crop = frame[max(0,y0+dy):min(h,y1-dy), max(0,x0+dx):min(w,x1-dx)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[8,6,6],[0,180,0,256,0,256])
    return cv2.normalize(hist,None,alpha=1,norm_type=cv2.NORM_L1).flatten().astype('float32')

class TargetLock:
    def __init__(self, seed, max_gap=0.5):
        self.box = np.asarray(seed,dtype=float)
        self.max_gap = max_gap
        self.last_time = None
        self.template = None
        self.expired = False
        self.ambiguous_count = 0
        self.reacquisitions = 0
        self.was_lost = False

    def update(self, frame, boxes, time):
        if self.expired:
            return None, 'reselect_required', np.nan
        if self.template is None:
            self.template = appearance(frame,self.box)
        if self.last_time is not None and time-self.last_time > self.max_gap + 1e-6:
            self.expired = True
            return None, 'reselect_required', np.nan
        if not len(boxes):
            self.was_lost = True
            return None, 'lost', np.nan
        ranked = []
        diagonal = max(np.linalg.norm(self.box[2:]-self.box[:2]), 40)
        dt = 0.1 if self.last_time is None else max(time-self.last_time,0.01)
        for b in boxes:
            b=np.asarray(b,dtype=float)
            dist = np.linalg.norm((b[:2]+b[2:]-self.box[:2]-self.box[2:])/2)/diagonal
            hist=appearance(frame,b)
            color = float(cv2.compareHist(self.template,hist,cv2.HISTCMP_BHATTACHARYYA)) if hist is not None and self.template is not None else 1.0
            overlap=iou(self.box,b)
            if dist > min(0.8,0.22 + 2*dt) or color > 0.65:
                continue
            cost=0.45*(1-overlap)+0.35*color+0.20*dist
            ranked.append((cost,b,color))
        ranked.sort(key=lambda item:item[0])
        if not ranked or ranked[0][0] > 0.58:
            self.was_lost=True
            return None,'lost',np.nan
        if len(ranked)>1 and ranked[1][0]-ranked[0][0] < 0.10:
            self.ambiguous_count+=1
            self.was_lost=True
            return None,'ambiguous',ranked[0][0]
        cost,box,color=ranked[0]
        if self.was_lost:
            self.reacquisitions+=1
        self.was_lost=False
        self.box=box
        self.last_time=time
        # Keep initial appearance fixed to avoid gradual identity drift.
        return box.copy(),'locked',cost
