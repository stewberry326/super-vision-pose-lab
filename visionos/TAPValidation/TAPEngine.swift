import CoreML
import CoreGraphics

/// Single-stream runner. Uses explicit recurrent memory, not hidden global state.
final class TAPEngine {
    private let features: MLModel
    private let query: MLModel
    private let update: MLModel
    private var queryLow: MLMultiArray?
    private var queryHigh: MLMultiArray?
    private var memory: MLMultiArray
    let points = 3
    init(computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration(); config.computeUnits = computeUnits
        func load(_ name: String) throws -> MLModel {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                throw NSError(domain: "TAP", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(name).mlmodelc가 없습니다. Core ML 변환 결과를 확인해주세요."])
            }
            return try MLModel(contentsOf: url, configuration: config)
        }
        features = try load("features"); query = try load("query"); update = try load("update")
        memory = try MLMultiArray(shape: [4,1,3,2,30720], dataType: .float32)
        memory.resetZeros()
    }
    func process(image: CGImage, seeds: [CGPoint]? = nil) throws -> (tracks: [CGPoint], confidence: [Double]) {
        let frame = try Self.frameTensor(image)
        let f = try features.prediction(from: MLDictionaryFeatureProvider(dictionary: ["frame": frame]))
        guard let low=f.featureValue(for: "low")?.multiArrayValue,
              let high=f.featureValue(for: "high")?.multiArrayValue else { throw failure("특징 출력 오류") }
        if let seeds {
            guard seeds.count==points else { throw failure("시작점은 3개여야 합니다.") }
            let p=try MLMultiArray(shape:[1,3,2],dataType:.float32)
            for (i,point) in seeds.enumerated() { p[i*2]=NSNumber(value:Double(point.x));p[i*2+1]=NSNumber(value:Double(point.y)) }
            let q=try query.prediction(from: MLDictionaryFeatureProvider(dictionary:["low":low,"high":high,"points":p]))
            queryLow=q.featureValue(for:"query_low")?.multiArrayValue
            queryHigh=q.featureValue(for:"query_high")?.multiArrayValue
            memory.resetZeros()
        }
        guard let queryLow,let queryHigh else { throw failure("먼저 시작점을 지정해주세요.") }
        let r=try update.prediction(from: MLDictionaryFeatureProvider(dictionary:["low":low,"high":high,"query_low":queryLow,"query_high":queryHigh,"memory":memory]))
        guard let xy=r.featureValue(for:"tracks")?.multiArrayValue,
              let occ=r.featureValue(for:"occlusion")?.multiArrayValue,
              let unc=r.featureValue(for:"uncertainty")?.multiArrayValue,
              let next=r.featureValue(for:"next_memory")?.multiArrayValue else { throw failure("추적 출력 오류") }
        memory=next
        return ((0..<points).map { CGPoint(x:xy[$0*2].doubleValue,y:xy[$0*2+1].doubleValue) },
                (0..<points).map { (1-occ[$0].doubleValue)*(1-unc[$0].doubleValue) })
    }
    private func failure(_ message:String)->NSError { NSError(domain:"TAP",code:2,userInfo:[NSLocalizedDescriptionKey:message]) }
    static func frameTensor(_ image:CGImage)throws->MLMultiArray {
        var rgba=[UInt8](repeating:0,count:256*256*4)
        let ok=rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let context=CGContext(data:bytes.baseAddress,width:256,height:256,bitsPerComponent:8,bytesPerRow:256*4,
                space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.interpolationQuality = .medium
            context.draw(image,in:CGRect(x:0,y:0,width:256,height:256));return true
        }
        guard ok else { throw NSError(domain:"TAP",code:3,userInfo:[NSLocalizedDescriptionKey:"이미지 전처리 실패"]) }
        let tensor=try MLMultiArray(shape:[1,1,256,256,3],dataType:.float32)
        let values=tensor.dataPointer.assumingMemoryBound(to:Float.self)
        for i in 0..<256*256 { for c in 0..<3 { values[i*3+c]=Float(rgba[i*4+c])/127.5-1 } }
        return tensor
    }
}
extension MLMultiArray {
    func resetZeros() { dataPointer.initializeMemory(as:UInt8.self,repeating:0,count:count*4) }
}
