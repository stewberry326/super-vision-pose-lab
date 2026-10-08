import SwiftUI
import AVFoundation
import UniformTypeIdentifiers
import CoreImage
import CoreML

@MainActor final class ValidationState: ObservableObject {
    @Published var image: UIImage?
    @Published var seeds: [CGPoint] = []
    @Published var tracks: [CGPoint] = []
    @Published var scores: [Double] = []
    @Published var status = "녹화 영상을 불러온 후 관절 위치 3개를 지정하세요."
    @Published var busy = false
    @Published var output: URL?
    var input: URL?
    func load(_ url: URL) {
        guard !busy else {return}
        _ = input?.stopAccessingSecurityScopedResource()
        _ = url.startAccessingSecurityScopedResource();input=url;seeds=[];tracks=[];output=nil
        Task {
            do {
                let generator=AVAssetImageGenerator(asset:AVURLAsset(url:url))
                generator.appliesPreferredTrackTransform=true
                let result=try await generator.image(at:.zero)
                image=UIImage(cgImage:result.image);status="관절 위치 3개를 지정하세요. 입력 영상은 외부로 전송하지 않습니다."
            } catch {status=error.localizedDescription}
        }
    }
    func run(cpuOnly:Bool) {
        guard let input,seeds.count==3,!busy else {return}
        busy=true;status="모델 로딩 · 프레임별 분석 중";output=nil
        let initial=seeds
        Task.detached {
            do {
                let engine=try TAPEngine(computeUnits:cpuOnly ? .cpuOnly : .all)
                let asset=AVURLAsset(url:input)
                guard let track=try await asset.loadTracks(withMediaType:.video).first else {throw NSError(domain:"Video",code:1)}
                let transform=try await track.load(.preferredTransform)
                let reader=try AVAssetReader(asset:asset)
                let video=AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
                video.alwaysCopiesSampleData=false;reader.add(video)
                guard reader.startReading() else {throw reader.error ?? NSError(domain:"Video",code:2)}
                let context=CIContext();var last = -Double.infinity;var first=true;var latency:[Double]=[]
                var csv="time_seconds,point,x_256,y_256,confidence,processing_ms\n"
                while let sample=video.copyNextSampleBuffer() {
                    let t=CMSampleBufferGetPresentationTimeStamp(sample).seconds
                    if t-last < 0.099 {continue};last=t
                    guard let pixel=CMSampleBufferGetImageBuffer(sample) else {continue}
                    let ci=CIImage(cvPixelBuffer:pixel).transformed(by:transform)
                    guard let cg=context.createCGImage(ci,from:ci.extent) else {continue}
                    let started=CFAbsoluteTimeGetCurrent()
                    let result=try engine.process(image:cg,seeds:first ? initial : nil);first=false
                    let ms=(CFAbsoluteTimeGetCurrent()-started)*1000;latency.append(ms)
                    for j in 0..<3 {csv += "\(t),\(j),\(result.tracks[j].x),\(result.tracks[j].y),\(result.confidence[j]),\(ms)\n"}
                    await MainActor.run {
                        self.image=UIImage(cgImage:cg);self.tracks=result.tracks;self.scores=result.confidence
                        self.status=String(format:"%.2f초 · %d 프레임 · %.1f ms",t,latency.count,ms)
                    }
                }
                if reader.status == .failed {throw reader.error ?? NSError(domain:"Video",code:3)}
                let url=FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0].appendingPathComponent("tap-\(Int(Date().timeIntervalSince1970)).csv")
                try csv.write(to:url,atomically:true,encoding:.utf8)
                let sorted=latency.sorted();let mean=latency.reduce(0,+)/Double(max(1,latency.count))
                let p95=sorted.isEmpty ? 0 : sorted[min(sorted.count-1,Int(Double(sorted.count)*0.95))]
                await MainActor.run {self.output=url;self.busy=false;self.status=String(format:"완료 %d 프레임 · 평균 %.1f / P95 %.1f ms. 녹화 영상 검증이며 실시간 성능을 뜻하지 않습니다.",latency.count,mean,p95)}
            } catch {await MainActor.run {self.status=error.localizedDescription;self.busy=false}}
        }
    }
}
struct ContentView: View {
    @StateObject private var state=ValidationState()
    @State private var importing=false
    @State private var cpuOnly=true
    var body: some View {
        VStack(alignment:.leading,spacing:16) {
            Text("온라인 TAP · Core ML 실행 검증").font(.title)
            Text("3개 표면점 추적 샘플 · 관절 검출과 임상 정확도 검증은 별도").foregroundStyle(.secondary)
            HStack {
                Button("녹화 영상 불러오기"){importing=true}.disabled(state.busy)
                Button("시작점 다시 지정"){state.seeds=[];state.tracks=[]}.disabled(state.busy)
                Toggle("CPU만 사용",isOn:$cpuOnly)
                Button("분석 시작"){state.run(cpuOnly:cpuOnly)}.disabled(state.seeds.count != 3 || state.busy)
            }
            if let image=state.image {
                ZStack(alignment:.topLeading) {
                    Image(uiImage:image).resizable().frame(width:400,height:400)
                    let points=state.tracks.isEmpty ? state.seeds : state.tracks
                    ForEach(Array(points.enumerated()),id:\.offset) { i,p in
                        if state.tracks.isEmpty || state.scores[i]>=0.5 {
                            Circle().fill(state.tracks.isEmpty ? Color.purple : Color.cyan).frame(width:10,height:10).position(x:p.x/256*400,y:p.y/256*400)
                        }
                    }
                }
                .frame(width:400,height:400).contentShape(Rectangle())
                .gesture(SpatialTapGesture().onEnded { event in
                    guard !state.busy,state.seeds.count<3 else {return}
                    state.seeds.append(CGPoint(x:event.location.x/400*256,y:event.location.y/400*256))
                })
                Text("처리 좌표와 맞추기 위해 미리보기를 정사각형으로 표시합니다. 지정 \(state.seeds.count)/3")
            }
            Text(state.status).font(.callout)
            if let output=state.output {ShareLink("좌표·지연 CSV 내보내기",item:output)}
        }.padding(28).frame(minWidth:700,minHeight:650)
        .fileImporter(isPresented:$importing,allowedContentTypes:[.movie]) { result in
            if case .success(let url)=result {state.load(url)}
        }
    }
}
