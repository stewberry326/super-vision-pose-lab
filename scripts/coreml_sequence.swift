import Foundation
import CoreML
let root=URL(fileURLWithPath:CommandLine.arguments[1])
let config=MLModelConfiguration();config.computeUnits = .cpuOnly
func load(_ name:String)throws->MLModel {try MLModel(contentsOf:root.appendingPathComponent("compiled/\(name).mlmodelc"),configuration:config)}
func array(_ shape:[NSNumber],_ values:[Float]?=nil)throws->MLMultiArray {
 let a=try MLMultiArray(shape:shape,dataType:.float32)
 let p=a.dataPointer.assumingMemoryBound(to:Float.self)
 for i in 0..<a.count {p[i]=values?[i] ?? 0};return a
}
func predict(_ model:MLModel,_ values:[String:MLMultiArray])throws->MLFeatureProvider {
 try model.prediction(from:MLDictionaryFeatureProvider(dictionary:values))
}
func get(_ r:MLFeatureProvider,_ key:String)->MLMultiArray {r.featureValue(for:key)!.multiArrayValue!}
func floats(_ a:MLMultiArray)->[Float] {(0..<a.count).map{a[$0].floatValue}}
let feature=try load("features"),query=try load("query"),update=try load("update")
let ref=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("sequence-reference.json"))) as! [[String:[Double]]]
var memory=try array([4,1,3,2,30720]);var qLow:MLMultiArray?;var qHigh:MLMultiArray?
var rows:[[String:Any]]=[]
for i in 0..<ref.count {
 let bytes=try Data(contentsOf:root.appendingPathComponent("frame-\(i).bin"))
 let input=try array([1,1,256,256,3]);bytes.withUnsafeBytes {src in input.dataPointer.copyMemory(from:src.baseAddress!,byteCount:bytes.count)}
 let start=Date()
 let f=try predict(feature,["frame":input]);let low=get(f,"low"),high=get(f,"high")
 if i==0 {
  let points=try array([1,3,2],[120,140,140,160,150,190])
  let q=try predict(query,["low":low,"high":high,"points":points]);qLow=get(q,"query_low");qHigh=get(q,"query_high")
 }
 let result=try predict(update,["low":low,"high":high,"query_low":qLow!,"query_high":qHigh!,"memory":memory])
 memory=get(result,"next_memory")
 var row:[String:Any]=["frame":i,"processing_ms":Date().timeIntervalSince(start)*1000]
 for key in ["tracks","occlusion","uncertainty"] {
  let pred=floats(get(result,key));row[key+"_max_abs_error"]=zip(pred,ref[i][key]!).map{abs(Double($0)-$1)}.max() ?? 0
 }
 rows.append(row)
}
let output:[String:Any]=["backend":"macOS Swift Core ML CPU_ONLY","sequence_frames":rows.count,"points":3,"resolution":256,"parity":rows,"vision_pro_device_test":false]
let json=try JSONSerialization.data(withJSONObject:output,options:[.prettyPrinted,.sortedKeys])
try json.write(to:root.appendingPathComponent("sequence-parity.json"));print(String(data:json,encoding:.utf8)!)
