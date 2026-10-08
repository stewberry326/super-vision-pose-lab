from pathlib import Path
import io
import os
import json
import zipfile
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from posecheck.schema import Settings,NAMES,LABELS
from posecheck.models import ROOT,Detector,load_model
from posecheck.video import probe,frame_at
from posecheck.pipeline import analyze
from posecheck.render import overlay
from posecheck.annotations import evaluate,compare,delay_report
from posecheck.tap_pipeline import run_tap,TAPSettings
from posecheck.reprocess import reprocess,review_split

st.set_page_config(page_title='Super Vision · Pose Lab',page_icon='◉',layout='wide')
st.markdown('''<style>
.stApp {background:#f6f8fb;color:#14263c;} [data-testid="stSidebar"] {background:#edf2f7;}
h1 {letter-spacing:-1.5px;} .eyebrow {color:#087f8c;font-weight:700;font-size:13px;letter-spacing:2px;}
div[data-testid="stMetric"] {background:white;border:1px solid #dce5ef;border-radius:12px;padding:16px;}
.stButton>button[kind="primary"] {background:#087f8c;border:0;} .block-container {padding-top:2rem;}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">SUPER VISION / TECHNICAL VALIDATION</div>',unsafe_allow_html=True)
st.title('부분 신체 관절 인식 검증')
st.caption('환자를 선택하고, 가까이 촬영한 영상에서 관절 인식과 보정 결과를 비교합니다. 모든 분석은 로컬에서 실행됩니다.')

@st.cache_data(show_spinner=False)
def cached_probe(path,mtime):return probe(path)

@st.cache_resource(show_spinner='사람 검출 모델을 준비하고 있습니다…')
def get_detector():return Detector()

@st.cache_resource(show_spinner='관절 모델을 준비하고 있습니다…')
def get_model(name,flip):return load_model(name,flip)

def get_meta(path):
    p=Path(path).expanduser()
    return cached_probe(str(p.resolve()),p.stat().st_mtime_ns)

def rgb_small(frame,width=960):
    scale=min(1,width/frame.shape[1]);display=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
    return cv2.cvtColor(display,cv2.COLOR_BGR2RGB),scale

with st.sidebar:
    st.subheader('검증 범위')
    st.write('앉았다 일어서기 · 근접 촬영')
    st.caption('영상 앞뒤를 제외한 14–34초를 예시 구간으로 준비했습니다. 실제 분석 범위는 직접 조절할 수 있습니다.')
    st.divider()
    st.markdown('**모델별 차이**')
    st.caption('ProbPose: 관절 품질·존재 확률·가시성 별도 출력\n\nRTMW: 133개 지점. 점수는 가시성 확률이 아닙니다.')
    st.markdown('[ProbPose 자료](https://huggingface.co/vrg-prague/ProbPose-s) · [RTMW 공식 코드](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)')
    st.divider()
    st.caption('2D 영상상 분석입니다. 카메라 기울기와 원근의 영향을 받으며, 임상 정확도와 Vision Pro 실시간 성능은 아직 검증하지 않았습니다.')

tab_input,tab_results,tab_manual,tab_tap,tab_quality=st.tabs(['① 영상 · 환자 선택','② 결과 · 모델 비교','③ 수동 기준점 검증','④ TAP 결합 · Core ML','⑤ 보정 규칙 · 전체 검토'])

with tab_input:
    st.subheader('분석할 영상')
    path=st.text_input('로컬 영상 경로',value=os.environ.get('POSE_TEST_VIDEO',''))
    upload=st.file_uploader('또는 영상 업로드',type=['mp4','mov'])
    if upload is not None:
        target=ROOT/'uploads'/Path(upload.name).name;target.parent.mkdir(exist_ok=True)
        signature=(upload.name,upload.size,upload.file_id)
        if st.session_state.get('upload_signature')!=signature:
            target.write_bytes(upload.getbuffer());st.session_state['upload_signature']=signature
        path=str(target)
    try:
        meta=get_meta(path)
    except Exception as exc:
        meta=None;st.error(f'영상 정보를 읽지 못했습니다: {exc}')
    if meta:
        st.caption(f"{meta['width']} × {meta['height']} · {meta['duration']:.2f}초 · {meta['frames']:,} 프레임 · 실제 타임스탬프 사용")
        input_key=str(Path(path).resolve())
        default_range=(14.0,34.0) if meta['duration']>34 else (meta['duration']*0.25,meta['duration']*0.80)
        start,end=st.slider('앞뒤를 제외할 분석 구간 (원본 영상의 초)',0.0,float(meta['duration']),
                            default_range,step=0.1,key='range_'+input_key)
        if start==end:st.error('분석 구간을 늘려주세요.')
        preview_t=st.slider('환자 선택 프레임',float(start),float(max(start+0.01,end)),float(start),step=0.1,key='preview_'+input_key)
        _,preview_time,frame=frame_at(meta,preview_t)
        if st.button('이 프레임에서 사람 찾기',type='primary'):
            with st.spinner('선택 가능한 사람을 찾고 있습니다…'):
                st.session_state['detection']={'path':input_key,'time':preview_time,'boxes':get_detector()(frame).tolist()}
        det=st.session_state.get('detection')
        seed_record=st.session_state.get('seed')
        if det and det['path']==input_key and abs(det['time']-preview_time)<0.05:
            boxes=np.array(det['boxes']).reshape(-1,4)
            disp,scale=rgb_small(frame)
            for i,b in enumerate(boxes):
                q=(b*scale).astype(int);cv2.rectangle(disp,tuple(q[:2]),tuple(q[2:]),(0,170,190),2)
                cv2.putText(disp,f'#{i+1}',tuple(q[:2]+[8,25]),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,170,190),2)
            st.write('분석할 사람을 클릭하세요. 검출이 안 되면 아래에서 영역을 직접 지정할 수 있습니다.')
            click=streamlit_image_coordinates(disp,key=f'seed_click_{input_key}_{preview_time}')
            if click:
                x,y=click['x']/scale,click['y']/scale
                hits=[(np.prod(b[2:]-b[:2]),b) for b in boxes if b[0]<=x<=b[2] and b[1]<=y<=b[3]]
                if hits:
                    b=min(hits,key=lambda z:z[0])[1]
                    st.session_state['seed']={'path':input_key,'time':preview_time,'box':b.tolist()};seed_record=st.session_state['seed']
            if len(boxes):
                number=st.selectbox('번호로 선택',list(range(1,len(boxes)+1)))
                if st.button('이 사람 록온'):
                    st.session_state['seed']={'path':input_key,'time':preview_time,'box':boxes[number-1].tolist()};seed_record=st.session_state['seed']
        else:
            st.image(rgb_small(frame)[0],width='stretch')
        with st.expander('검출이 어려운 부분 신체: 영역 직접 지정'):
            st.caption('선택 영역은 초기 록온 기준입니다. 이후 검출·동일인 연결에 실패하면 추적 유실로 처리합니다.')
            c1,c2,c3,c4=st.columns(4)
            x0=c1.number_input('왼쪽 x',0,meta['width']-1,value=int(meta['width']*0.20))
            y0=c2.number_input('위쪽 y',0,meta['height']-1,value=int(meta['height']*0.20))
            x1=c3.number_input('오른쪽 x',1,meta['width'],value=int(meta['width']*0.65))
            y1=c4.number_input('아래쪽 y',1,meta['height'],value=meta['height'])
            if st.button('직접 지정 영역 사용'):
                if x1>x0 and y1>y0:
                    st.session_state['seed']={'path':input_key,'time':preview_time,'box':[x0,y0,x1,y1]};seed_record=st.session_state['seed']
                else:st.error('영역의 크기가 0보다 커야 합니다.')
        if seed_record and seed_record['path']==input_key:
            st.success(f"대상 선택됨 · {seed_record['time']:.2f}초 · {np.round(seed_record['box']).astype(int).tolist()}")
            st.caption('분석은 대상 선택 시점부터 시작됩니다. 앞의 동작도 보려면 구간 시작 프레임에서 다시 선택해주세요.')
        model_names=st.multiselect('비교할 모델',['ProbPose-s','RTMW-l'],default=['ProbPose-s','RTMW-l'])
        condition=st.selectbox('촬영 조건',['근접·부분 신체','전신','상반신','하반신','손·몸에 의한 가림','카메라 움직임'])
        with st.expander('분석·보정 설정',expanded=False):
            c1,c2,c3=st.columns(3)
            fps=c1.number_input('분석 빈도 (프레임/초)',1.0,30.0,10.0,step=1.0)
            threshold=c2.slider('관절 품질 기준',0.0,1.0,0.35,0.05)
            max_gap=c3.slider('최대 보간 간격 (앞뒤 유효점 사이 초)',0.0,1.0,0.25,0.05)
            c1,c2,c3=st.columns(3)
            presence=c1.slider('ProbPose 존재 확률 기준',0.0,1.0,0.50,0.05)
            visibility=c2.slider('ProbPose 가시성 기준',0.0,1.0,0.50,0.05)
            tau=c3.slider('평활화 시간 상수 (초, 0=끄기)',0.0,0.3,0.06,0.01)
            c1,c2,c3=st.columns(3)
            speed=c1.number_input('좌표 이동 제한 (몸 크기/초)',0.5,30.0,5.0,step=0.5)
            bone=c2.slider('분절 길이 변화 허용 비율',0.2,2.0,0.65,0.05)
            track_gap=c3.slider('자동 재연결 허용 유실 시간 (초)',0.1,2.0,0.5,0.1)
            roll=st.number_input('수동 카메라 기울기 보정 (도)',-180.0,180.0,0.0)
            flip=st.checkbox('ProbPose 공식 설정의 좌우 반전 평균 추론',value=True)
            st.caption('카메라 회전은 자동 보정하지 않습니다. 강한 평활화는 신전 속도와 발생 시점을 바꿀 수 있습니다.')
        ready=bool(model_names and seed_record and seed_record['path']==input_key and start<=seed_record['time']<end)
        if st.button('선택 구간 분석하기',type='primary',disabled=not ready):
            try:
                bar=st.progress(0,text='모델 준비')
                settings=Settings(start=seed_record['time'],end=end,sample_fps=fps,confidence=threshold,
                                  presence=presence,visibility=visibility,max_gap_seconds=max_gap,
                                  max_speed_body_per_second=speed,bone_change_ratio=bone,smoothing_tau=tau,
                                  max_track_gap=track_gap,camera_roll_degrees=roll,condition=condition,flip_test=flip)
                loaded={};load_errors={}
                for name in model_names:
                    try:loaded[name]=get_model(name,flip)
                    except Exception as exc:load_errors[name]=str(exc);st.error(f'{name} 준비 실패: {exc}')
                if not loaded:raise RuntimeError('실행 가능한 모델이 없습니다.')
                out=analyze(path,seed_record['box'],settings,list(loaded),
                            progress=lambda p,message:bar.progress(p,text=message),model_cache=loaded,detector=get_detector())
                if load_errors:
                    report=json.loads((out/'report.json').read_text());report['model_load_errors'].update(load_errors)
                    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                st.session_state['latest_run']=str(out);st.success('분석 완료. 결과 탭에서 확인하세요.')
            except Exception as exc:st.exception(exc)

runs=sorted((ROOT/'outputs').glob('*/report.json'),reverse=True)
run=None;report=None
with tab_results:
    if not runs:st.info('먼저 영상에서 환자를 선택하고 분석해주세요.')
    else:
        choices=[str(p.parent) for p in runs]
        latest=st.session_state.get('latest_run',choices[0])
        run=Path(st.selectbox('저장된 분석',choices,index=choices.index(latest) if latest in choices else 0,format_func=lambda x:Path(x).name))
        report=json.loads((run/'report.json').read_text())
        st.caption(f"원본 {report['settings']['start']:.2f}–{report['settings']['end']:.2f}초 · {report['sampled_frames']} 분석 프레임 · {report['settings']['condition']}")
        if report['model_load_errors']:st.error('모델 준비 실패: '+str(report['model_load_errors']))
        c1,c2,c3=st.columns(3)
        c1.metric('추적 유실',f"{report['track_lost_seconds']:.2f}초")
        c2.metric('대상 연결 모호함',str(report['association_ambiguities']))
        c3.metric('전체 처리 시간',f"{report['processing_seconds']:.1f}초")
        st.caption('록온 비율은 추적기의 판단입니다. 동일인을 정확하게 유지했는지는 영상으로 검수해야 합니다.')
        if 'quality_experiment' in report:
            st.caption('보정 v2 실험: 처리 시간은 저장된 RAW 재처리·영상 생성 시간입니다. 노란 점/점선은 불확실성 표시이며, 해당 점이나 연결선 경고가 있는 각도는 측정에서 제외합니다.')
        chosen=st.multiselect('화면에서 비교할 모델',list(report['models']),default=[n for n in report['models'] if n!='RTMW-l'] if 'ProbPose-TAP' in report['models'] else list(report['models']),key='result_models_'+str(run))
        metric=st.selectbox('그래프 지표',['left_knee','right_knee','trunk','left_elbow','right_elbow','left_knee_velocity','right_knee_velocity'],
                            format_func=lambda x:{'left_knee':'왼무릎 끼인각','right_knee':'오른무릎 끼인각','trunk':'영상 수직선 대비 체간 기울기',
                            'left_elbow':'왼팔꿈치 끼인각','right_elbow':'오른팔꿈치 끼인각','left_knee_velocity':'왼무릎 각속도','right_knee_velocity':'오른무릎 각속도'}[x])
        fig=go.Figure();tables={}
        colors=['#078b99','#8356c7','#db6b28','#4768a0']
        for k,name in enumerate(chosen):
            table=pd.read_csv(run/name/'metrics.csv');tables[name]=table
            base=metric.replace('_velocity','')
            suffix='_deg_per_sec' if metric.endswith('_velocity') else '_deg'
            for variant in (['corrected'] if name in ['ProbPose-TAP','ProbPose-v2'] else ['raw','corrected']):
                fig.add_trace(go.Scatter(x=table.time_seconds,y=table[base+'_'+variant+suffix],mode='lines',
                              name=name+' · '+('원본' if variant=='raw' else ('선택 결과' if name=='ProbPose-TAP' else '보정')),connectgaps=False,
                              line={'color':colors[k%len(colors)],'dash':'dot' if variant=='raw' else 'solid','width':1 if variant=='raw' else 2}))
            state_col=base+'_velocity_state' if metric.endswith('_velocity') else base+'_state'
            mask=table[state_col]=='interpolated_influence'
            if mask.any():
                fig.add_trace(go.Scatter(x=table.time_seconds[mask],y=table.loc[mask,base+'_corrected'+suffix],
                                        mode='markers',name=name+' · 보간 영향',marker={'color':'#efa12a','size':6}))
        fig.update_layout(height=390,margin={'l':10,'r':10,'t':15,'b':110},plot_bgcolor='white',paper_bgcolor='white',
                          xaxis_title='원본 영상 시각 (초)',yaxis_title='도/초' if metric.endswith('_velocity') else '도',
                          legend={'orientation':'h','yanchor':'top','y':-.30,'x':0})
        event=st.plotly_chart(fig,width='stretch',on_select='rerun',selection_mode='points',key='chart_'+str(run)+'_'+metric)
        frame_time=st.slider('같은 시점의 관절 비교',float(report['settings']['start']),float(report['settings']['end']),
                             float(report['settings']['start']),step=0.1,key='cursor_'+str(run))
        if event.selection.points:frame_time=float(event.selection.points[-1]['x'])
        display_mode=st.radio('관절 표시',['보정','측정용','원본'] if 'ProbPose-v2' in report['models'] else ['보정','원본'],horizontal=True)
        source_meta=get_meta(report['source']['path'])
        if chosen:
            cols=st.columns(len(chosen))
            for col,name in zip(cols,chosen):
                with col:
                    st.markdown('**'+name+'**')
                    summary=report['models'][name]
                    st.caption(f"왼무릎 원본 유효 {summary['raw_valid_rates']['left_knee']*100:.1f}% · 오른무릎 {summary['raw_valid_rates']['right_knee']*100:.1f}% · 보간 {summary['interpolation_ratio_body']*100:.1f}%")
                    data=np.load(run/name/'arrays.npz');i=int(np.argmin(abs(data['times']-frame_time)))
                    _,_,f=frame_at(source_meta,float(data['times'][i]))
                    box=data['boxes'][i];box=None if not np.isfinite(box).all() else box
                    warnings=None
                    if display_mode=='원본':points=data['raw'][i];states=np.full(23,'observed')
                    elif display_mode=='보정' and 'display' in data:
                        points=data['display'][i];states=data['display_states'][i];warnings=data['edge_warnings'][i]
                    else:
                        points=data['corrected'][i];states=data['states'][i]
                        if 'edge_warnings' in data:warnings=data['edge_warnings'][i]
                    state=tables[name].track_state.iloc[i]
                    st.image(cv2.cvtColor(overlay(f,points,states,box,data['times'][i],state,name,width=960,edge_warnings=warnings),cv2.COLOR_BGR2RGB),width='stretch')
                    st.caption('환자의 해부학적 좌우입니다. 파랑=TAP 추적, 보라=재검출, 주황=보간 영향. 점이 없으면 측정 불가입니다.')
                    if name=='ProbPose-TAP':st.caption('비교 영상: 왼쪽 ProbPose 원본 · 가운데 기존 보정 · 오른쪽 TAP 결합')
                    if name=='ProbPose-v2':st.caption('비교 영상: RAW / 기존 보정 / v2 표시. 노란 점·점선은 확정 좌표가 아닙니다. ⑤ 탭에서 관절별 이유를 확인하세요.')
                    st.video(str(run/name/'comparison.mp4'))
                    st.download_button('좌표 CSV',data=(run/name/'coordinates.csv').read_bytes(),file_name=name+'_coordinates.csv',key='csv_'+name)
                    st.download_button('각도·속도 CSV',data=(run/name/'metrics.csv').read_bytes(),file_name=name+'_metrics.csv',key='metric_'+name)
        with st.expander('원본 구간 · 설정 · 제한사항'):
            st.video(str(run/'source_excerpt.mp4'))
            st.json(report)
        if st.button('전체 결과 ZIP 준비'):
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
                for p in run.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(run))
            st.download_button('ZIP 다운로드',buffer.getvalue(),file_name=run.name+'.zip')

with tab_manual:
    st.subheader('영상상 수동 기준점과 비교')
    st.caption('직접 표시한 점은 영상상 비교 기준이며, 촉진한 관절축이나 임상 측정의 정답을 뜻하지 않습니다.')
    if run is None:st.info('분석을 완료하면 기준점을 표시할 수 있습니다.')
    else:
        name=st.selectbox('주석할 모델',list(report['models']),key='manual_model')
        folder=run/name;data=np.load(folder/'arrays.npz')
        i=st.select_slider('주석할 분석 프레임',options=list(range(len(data['times']))),format_func=lambda j:f"{data['times'][j]:.3f}초",key='annot_frame_'+str(run))
        joint=st.selectbox('기준점',list(range(23)),format_func=lambda j:LABELS[j],key='annot_joint')
        _,_,frame=frame_at(get_meta(report['source']['path']),float(data['times'][i]))
        display,scale=rgb_small(frame)
        label_path=run/'annotations.csv'
        labels=pd.read_csv(label_path) if label_path.exists() else pd.DataFrame(columns=['frame_index','joint','x','y','visible','role','split','certainty','condition'])
        current=labels[labels.frame_index==int(data['indices'][i])]
        for r in current.itertuples():
            if r.visible and np.isfinite(r.x):
                p=(int(r.x*scale),int(r.y*scale));cv2.circle(display,p,5,(255,150,0),-1)
                cv2.putText(display,r.joint,p,cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,150,0),1)
        st.write('예측을 숨긴 원본 화면에서 관절 기준점을 클릭하면 저장됩니다. TAP 입력점은 ④ 탭에서 따로 지정하세요. 이전 버전 기준점은 최종 평가용으로 분류되므로 필요하면 용도를 수정해주세요.')
        c1,c2,c3=st.columns(3)
        prescribed=review_split(float(data['times'][i]),report['settings']['start'],report['quality_experiment'].get('known_examples',True)) if 'quality_experiment' in report else None
        split_options=['calibration'] if prescribed=='calibration' else ['evaluation','calibration']
        annot_split=c1.selectbox('기준점 용도',split_options,format_func=lambda x:'최종 평가용' if x=='evaluation' else '설정 조정용',key=f'annot_split_{run}_{i}')
        if prescribed=='calibration':st.caption('이미 살펴본 사례 또는 조정용 시간 블록입니다. 이 기준점은 최종 평가에 포함하지 않습니다.')
        certainty=c2.selectbox('기준점 확신',['high','medium','low'],format_func=lambda x:{'high':'높음','medium':'보통','low':'낮음'}[x])
        annot_condition=c3.selectbox('프레임 조건',['visible','occluded','truncated','moving_camera','unspecified'],format_func=lambda x:{'visible':'잘 보임','occluded':'가림','truncated':'화면 잘림','moving_camera':'카메라 이동','unspecified':'미분류'}[x])
        generation=st.session_state.get('annotation_generation',0)
        click=streamlit_image_coordinates(display,key=f'annot_{run}_{name}_{i}_{joint}_{generation}')
        action=None
        if click:action={'frame_index':int(data['indices'][i]),'joint':NAMES[joint],'x':click['x']/scale,'y':click['y']/scale,'visible':True}
        if st.button('이 관절은 보이지 않음'):
            action={'frame_index':int(data['indices'][i]),'joint':NAMES[joint],'x':np.nan,'y':np.nan,'visible':False}
        if action:
            action.update(role='evaluation',split=annot_split,certainty=certainty,condition=annot_condition)
            mask=(labels.frame_index==action['frame_index'])&(labels.joint==action['joint'])
            labels=pd.concat([labels[~mask],pd.DataFrame([action])],ignore_index=True)
            labels.to_csv(label_path,index=False)
            st.success('기준점 저장됨')
        if len(labels):
            errors,angles=evaluate(folder,annot_split)
            st.dataframe(labels,width='stretch',hide_index=True)
            st.markdown('**관절점 위치 오차 · 원본 해상도 픽셀**')
            if not errors.empty:st.dataframe(errors[['frame_index','joint','visible','raw_error_px','corrected_error_px','unseen_but_raw_accepted']],hide_index=True)
            else:st.info('이 구분의 평가용 기준점이 없거나 TAP 입력점과 겹쳐 평가에서 제외됐습니다.')
            if len(angles):
                st.markdown('**세 기준점이 있는 관절의 영상상 각도 비교**');st.dataframe(angles,hide_index=True)
            if st.button('현재 프레임·관절의 주석 삭제'):
                labels=labels[~((labels.frame_index==int(data['indices'][i]))&(labels.joint==NAMES[joint]))]
                labels.to_csv(label_path,index=False)
                st.session_state['annotation_generation']=generation+1
                st.rerun()

        st.divider()
        st.markdown('**같은 기준점으로 모든 방식 비교**')
        summary_split=st.selectbox('비교할 기준점 구분',['evaluation','calibration'],format_func=lambda x:'최종 평가용' if x=='evaluation' else '설정 조정용',key='summary_split')
        large_ratio=st.number_input('큰 오차 기준: 영상 대각선 대비 비율',.001,.20,.02,.005,format='%.3f')
        if st.button('정확도 비교표 생성'):
            summary,details=compare(run,summary_split,large_ratio)
            if summary.empty:st.info('평가용 기준점을 먼저 입력해주세요. 입력·재초기화에 사용한 점은 제외합니다.')
            else:
                st.dataframe(summary,hide_index=True,width='stretch')
                st.download_button('비교표 CSV',summary.to_csv(index=False).encode(),file_name='accuracy_summary.csv')
                st.caption('픽셀 오차와 제공률을 함께 보세요. 정규화 기준은 고정된 원본 영상 대각선입니다. 임상 정확도를 의미하지 않습니다.')
        if st.button('연속 기준점으로 지연 비교'):
            delays=delay_report(run,summary_split)
            if not delays.empty:st.dataframe(delays,hide_index=True)
            else:st.info('연속된 평가용 기준점이 필요합니다.')
        st.caption('추적 지연은 연속된 수동 주석이 있어야 비교할 수 있습니다. 드문 프레임의 기준점만으로 지연을 단정하지 않습니다.')

with tab_tap:
    st.subheader('ProbPose + 온라인 BootsTAPIR')
    st.write('기존 분석의 동일 프레임을 사용해 관절 재검출과 점 추적을 결합합니다. 미래 프레임을 사용하지 않고, 추가 보간·평활화도 적용하지 않습니다.')
    if run is None or 'ProbPose-s' not in report['models']:
        st.info('ProbPose-s 분석을 먼저 실행해주세요.')
    else:
        st.caption('검출 위치와 추적 위치가 모두 신뢰할 만하지만 크게 다르면 값을 평균내지 않고 불일치로 남깁니다. TAP 점수와 ProbPose 점수는 의미가 다릅니다.')
        c1,c2,c3=st.columns(3)
        interval=c1.number_input('관절 재검출 주기 (초)',.1,3.,.5,.1)
        tap_quality=c2.slider('TAP 유효 기준',0.,1.,.5,.05)
        difference=c3.slider('검출·추적 불일치 기준 (몸 크기 대비)',.01,.30,.06,.01)
        c1,c2,c3=st.columns(3)
        det_quality=c1.slider('재검출 관절 품질 기준',0.,1.,.6,.05)
        det_visibility=c2.slider('재검출 가시성 기준',0.,1.,.6,.05)
        device=c3.selectbox('컴퓨터 추론 장치',['mps','cpu'],format_func=lambda x:'Mac GPU (MPS)' if x=='mps' else 'CPU')
        tap_resolution=st.selectbox('TAP 입력 해상도',[256,512],format_func=lambda n:f'{n} × {n}')
        st.caption('512 입력은 추가 고해상도 refinement를 사용합니다. 정확도 개선은 별도 비교해야 하며, Core ML 최소 샘플은 256 입력입니다.')
        seed_key='tap_manual_'+str(run)
        if seed_key not in st.session_state:st.session_state[seed_key]=[]
        with st.expander('수동 추적 시작점 / 재초기화점 (평가용 기준점과 분리)'):
            seed_data=np.load(run/'ProbPose-s/arrays.npz')
            si=st.select_slider('TAP 입력 프레임',options=list(range(len(seed_data['times']))),format_func=lambda j:f"{seed_data['times'][j]:.3f}초",key='tap_seed_i_'+str(run))
            sj=st.selectbox('TAP 입력 관절',list(range(5,17)),format_func=lambda j:LABELS[j])
            sf=frame_at(get_meta(report['source']['path']),float(seed_data['times'][si]))[2]
            sd,sscale=rgb_small(sf)
            for point in st.session_state[seed_key]:
                if point['frame_index']==int(seed_data['indices'][si]):
                    cv2.circle(sd,(round(point['x']*sscale),round(point['y']*sscale)),6,(230,80,210),-1)
            generation=st.session_state.get('tap_seed_generation',0)
            sclick=streamlit_image_coordinates(sd,key=f'tap_input_{run}_{si}_{sj}_{generation}')
            if sclick:
                point={'frame_index':int(seed_data['indices'][si]),'joint':NAMES[sj],'x':sclick['x']/sscale,'y':sclick['y']/sscale}
                points=[p for p in st.session_state[seed_key] if (p['frame_index'],p['joint'])!=(point['frame_index'],point['joint'])]
                st.session_state[seed_key]=points+[point]
            st.write(st.session_state[seed_key])
            if st.button('현재 입력점 제거'):
                st.session_state[seed_key]=[p for p in st.session_state[seed_key] if (p['frame_index'],p['joint'])!=(int(seed_data['indices'][si]),NAMES[sj])]
                st.session_state['tap_seed_generation']=generation+1;st.rerun()
            st.caption('여기서 지정한 프레임·관절은 모든 비교 방식의 정확도 평가에서 제외합니다. 가려진 관절을 추측해 지정하지 마세요.')
        if st.button('이 분석에 TAP 결합 실행',type='primary'):
            try:
                bar=st.progress(0.,text='온라인 TAP 모델 준비')
                config=TAPSettings(interval,tap_quality,difference,det_quality,det_visibility,device,tap_resolution)
                output=run_tap(run,config,lambda f,m:bar.progress(f,text=m),st.session_state[seed_key])
                st.session_state['latest_run']=str(output);st.success('완료. ② 결과 탭에서 ProbPose-TAP을 비교하세요.');st.rerun()
            except Exception as exc:st.exception(exc)
        if 'hybrid' in report:
            st.json(report['hybrid'])
            st.caption('기존 분석의 검출 좌표를 재사용하므로, 처리 시간은 검출까지 포함한 Vision Pro 전체 지연이 아닙니다.')
    st.divider();st.subheader('Core ML 변환 실험')
    cmreport=ROOT/'coreml/report.json'
    if cmreport.exists():
        cm=json.loads(cmreport.read_text())
        st.success('Core ML 3개 모델 변환 · Mac 연속 프레임 실행 · visionOS 서명 없는 빌드 확인')
        st.caption('최소 샘플: 256 × 256 입력, 수동 지정한 3개 점. Vision Pro 실기기 실행과 전체 관절 검출 결합은 아직 확인하지 않았습니다.')
        st.json(cm)
        sequence_report=ROOT/'coreml/sequence-parity.json'
        if sequence_report.exists():st.json(json.loads(sequence_report.read_text()))
        st.caption('Mac 실행·변환 수치와 Vision Pro 실기기 검증은 구분합니다. 실기기 미검증 결과를 완료로 표시하지 않습니다.')
    else:st.info('Core ML 변환 실험 결과가 아직 없습니다.')

with tab_quality:
    st.subheader('보정 규칙 v2 · 전체 구간 검토')
    st.write('길이 변화만으로 관절을 삭제하지 않고, 공통 이동을 제외한 개별 관절 움직임을 함께 확인합니다. 기존 결과를 보존하고 동일 RAW에 새 규칙을 적용합니다.')
    if run is None or 'ProbPose-s' not in report['models']:
        st.info('ProbPose-s 분석이 필요합니다.')
    else:
        if st.button('이 분석의 RAW로 보정 v2 비교 만들기'):
            try:
                bar=st.progress(0.,text='보정 규칙 및 전체 검토 목록 준비')
                output=reprocess(run,lambda f,m:bar.progress(f,text=m))
                st.session_state['latest_run']=str(output);st.rerun()
            except Exception as exc:st.exception(exc)
        if 'quality_experiment' not in report:
            st.info('위 버튼으로 RAW / 기존 보정 / v2 비교와 전체 구간의 검토 목록을 만듭니다.')
        else:
            experiment=report['quality_experiment']
            st.caption(f"원본 {report['settings']['start']:.1f}–{report['settings']['end']:.1f}초 · {report['sampled_frames']}개 분석 프레임 전체 검사. 원본 영상 전체/모든 원본 프레임 검증은 아닙니다.")
            overview=json.loads((run/'review_summary.json').read_text())
            a,b,c=st.columns(3)
            a.metric('문제 후보 프레임',overview['candidate_frames'])
            b.metric('검토 표본',overview['queue_frames'])
            c.metric('정확도 검증','기준점 필요')
            st.caption('문제 후보는 오류 확정이 아닙니다. 위험 구간 대표·1초 간격·고정 난수 표본을 포함합니다. 시간 블록 분리는 같은 영상 내 비교이며, 새 영상 검증을 대신하지 않습니다.')
            queue=pd.read_csv(run/'review_queue.csv');all_frames=pd.read_csv(run/'review_frames.csv')
            queue_labels={'risk_window_peak':'위험 구간 대표','systematic_sample':'1초 간격','fixed_random_sample':'무작위 표본','user_reported_example':'사용자 제보'}
            qindex=st.selectbox('검토할 표본',list(range(len(queue))),format_func=lambda j:f"{queue.iloc[j].time_seconds:.2f}초 · "+' / '.join(queue_labels.get(x,x) for x in queue.iloc[j].selection_source.split(' | ')))
            sample=queue.iloc[qindex];frame_index=int(sample.frame_index);t=float(sample.time_seconds)
            st.caption(('설정 조정용' if sample.suggested_split=='calibration' else '최종 평가용으로 남겨둔 구간')+' · 예측 비교를 보고 기준점을 찍으면 판단이 영향을 받을 수 있으니, ③ 탭에서 원본만 보고 지정하세요.')
            st.image(str(run/'review'/f'{frame_index:06d}_comparison.jpg'),width='stretch',caption='왼쪽 RAW · 가운데 기존 보정 · 오른쪽 v2 표시 (노란 점/점선: 불확실)')
            with st.expander('원본만 보기'):
                st.image(str(run/'review'/f'{frame_index:06d}_original.jpg'),width='stretch')
            diag=pd.read_csv(run/'ProbPose-v2/coordinates.csv')
            diag=diag[(diag.frame_index==frame_index)&diag.joint.isin(NAMES[5:17])].copy()
            old=pd.read_csv(run/'ProbPose-s/coordinates.csv')
            old=old[old.frame_index==frame_index][['joint','rejection_reason','corrected_state']].rename(columns={'rejection_reason':'old_reason','corrected_state':'old_state'})
            diag=diag.merge(old,on='joint');diag['관절']=diag.joint.map(dict(zip(NAMES,LABELS)))
            reason_labels={'valid':'유효','outlier_bone':'길이 변화로 제외','outlier_speed':'속도 급변으로 제외','outlier_isolated_motion_and_segment':'개별 움직임+길이 이상으로 제외',
                'uncertain_motion':'개별 움직임 의심 (표시만)','low_score':'품질 낮음','low_visibility':'가시성 낮음','out_of_frame':'화면 밖','unsupported_or_missing':'좌표 없음','low_presence':'존재 가능성 낮음'}
            diag['기존 처리']=diag.old_reason.map(lambda x:reason_labels.get(x,x))
            diag['v2 처리']=diag.v2_reason.map(lambda x:reason_labels.get(x,x))
            diag['길이 경고 수']=diag.segment_warning_count
            diag['측정용 좌표']=diag.measurement_available
            st.dataframe(diag[['관절','기존 처리','v2 처리','길이 경고 수','측정용 좌표','score','visibility']],hide_index=True,width='stretch')
            st.caption('“유효”는 규칙 통과를 뜻합니다. 실제 관절 위치가 맞다는 정답 판정은 아닙니다. 좌표가 있어도 연결선 경고가 있는 각도는 제외될 수 있습니다.')
            def choose_annotation_frame():
                st.session_state['annot_frame_'+str(run)]=int(sample.sample_index)
            st.button('③ 기준점 검증의 프레임을 이 시점으로 맞추기',on_click=choose_annotation_frame)
            st.markdown('**원본 영상으로 판단 기록**')
            review_joint=st.selectbox('판단할 관절',list(range(5,17)),format_func=lambda j:LABELS[j],key='review_joint')
            judgments=['not_reviewed','correct','incorrect','unjudgeable']
            labels={'not_reviewed':'미검토','correct':'위치가 맞아 보임','incorrect':'위치가 틀려 보임','unjudgeable':'가림·잘림 등으로 판단 불가'}
            left,right=st.columns(2)
            raw_judgment=left.selectbox('RAW 위치 판단',judgments,format_func=labels.get,key=f'raw_review_{run}_{frame_index}_{review_joint}')
            v2_judgment=right.selectbox('v2 화면 위치 판단',judgments,format_func=labels.get,key=f'v2_review_{run}_{frame_index}_{review_joint}')
            comment=st.text_input('판단 근거 / 가림·카메라 움직임',key=f'review_comment_{run}_{frame_index}_{review_joint}')
            if st.button('이 판단 저장'):
                p=run/'review_judgments.csv'
                judgments_df=pd.read_csv(p) if p.exists() else pd.DataFrame(columns=['frame_index','joint','raw_judgment','v2_display_judgment','split','note'])
                item={'frame_index':frame_index,'joint':NAMES[review_joint],'raw_judgment':raw_judgment,'v2_display_judgment':v2_judgment,'split':sample.suggested_split,'note':comment}
                mask=(judgments_df.frame_index==frame_index)&(judgments_df.joint==item['joint'])
                pd.concat([judgments_df[~mask],pd.DataFrame([item])],ignore_index=True).to_csv(p,index=False)
                st.success('검토 판단을 저장했습니다. 정량 오차는 ③ 탭의 수동 기준점으로 비교하세요.')
            with st.expander('전체 프레임 후보 · 관절별 규칙 비교'):
                st.dataframe(all_frames,hide_index=True,width='stretch')
                st.dataframe(pd.read_csv(run/'rule_comparison.csv'),hide_index=True,width='stretch')
            for filename,label in [('review_queue.csv','검토 목록 CSV'),('review_frames.csv','전체 프레임 진단 CSV'),('rule_comparison.csv','관절별 규칙 비교 CSV')]:
                st.download_button(label,(run/filename).read_bytes(),file_name=filename)
            st.video(str(run/'ProbPose-v2/comparison.mp4'))
