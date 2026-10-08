from pathlib import Path
import cv2
import numpy as np
from .schema import EDGES

COLORS={'observed':(210,210,210),'smoothed':(190,220,40),'uncertain':(30,210,255),
        'tracked':(230,150,70),'detection':(80,220,120),'redetected':(180,70,230),'manual_anchor':(210,110,220),'interpolated':(30,170,255),'smoothed_after_interpolation':(50,180,240)}

def overlay(frame,points,states,box,time,track_state,title,width=1280,edge_warnings=None):
    factor=min(1.0,width/frame.shape[1]);w=round(frame.shape[1]*factor)
    h=round(frame.shape[0]*factor);h-=h%2;w-=w%2
    image=cv2.resize(frame,(w,h))
    pts=points*factor
    for e,(a,b) in enumerate(EDGES):
        if np.isfinite(pts[[a,b]]).all():
            caution = any(states[j]=='uncertain' for j in [a,b]) or (edge_warnings is not None and edge_warnings[e])
            color=(30,210,255) if caution else ((30,170,255) if any('interpol' in str(states[j]) for j in [a,b]) else (190,220,40))
            if caution:
                start,end=pts[a],pts[b];length=np.linalg.norm(end-start)
                if length>0:
                    for offset in np.arange(0,length,14):
                        p=start+(end-start)*offset/length;q=start+(end-start)*min(offset+8,length)/length
                        cv2.line(image,tuple(p.astype(int)),tuple(q.astype(int)),color,2,cv2.LINE_AA)
            else:cv2.line(image,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),color,2,cv2.LINE_AA)
    for j,point in enumerate(pts):
        if np.isfinite(point).all():
            color=COLORS.get(states[j],(110,110,255))
            cv2.circle(image,tuple(point.astype(int)),4,color,-1,cv2.LINE_AA)
    if box is not None:
        b=(box*factor).astype(int)
        cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),(220,190,30),2)
    cv2.rectangle(image,(0,0),(w,68),(24,29,36),-1)
    cv2.putText(image,f'{title} | source {time:.3f}s | {track_state}',(16,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(240,240,240),1,cv2.LINE_AA)
    legend = 'Cyan: accepted | Yellow dashed: uncertain (not measured) | Orange: interpolation' if edge_warnings is not None else 'Gray: raw | Cyan: filter | Blue: TAP | Purple: reset | Orange: interpolation'
    cv2.putText(image,legend,
                (16,51),cv2.FONT_HERSHEY_SIMPLEX,0.43,(190,190,190),1,cv2.LINE_AA)
    return image

def graph_png(table,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for axis,metric,title in zip(axes,['left_knee','right_knee','trunk'],
                                ['Left knee included angle (2D)','Right knee included angle (2D)','Trunk vs image vertical (2D)']):
        for variant,color in [('raw','#94a3b8'),('corrected','#0891b2')]:
            axis.plot(table.time_seconds,table[metric+'_'+variant+'_deg'],label=variant,color=color,lw=1)
        state=table[metric+'_state']=='interpolated_influence'
        axis.scatter(table.time_seconds[state],table.loc[state,metric+'_corrected_deg'],s=9,color='#e99c25',label='interpolation influence')
        axis.set_ylabel('degrees');axis.set_title(title,loc='left',fontsize=10);axis.grid(alpha=0.2);axis.legend(fontsize=8)
    axes[-1].set_xlabel('Original video time (seconds)')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
