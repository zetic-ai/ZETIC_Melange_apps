import CoreGraphics
import Foundation
import ZeticMLange

struct Detection {
    let rect: CGRect        // normalized 0..1 in frame coordinates
    let score: Float
    let classIndex: Int
}

/// Decodes YOLO26n output. Handles both layouts (auto-detected by shape, following
/// the YOLO26 reference app): the NMS-free export [1,300,6] with rows
/// (x1,y1,x2,y2,score,class) in 0..640 pixels, and the raw [1,4+C,anchors] head.
enum DetectorDecoder {
    static func decode(_ output: Tensor) -> (person: Detection?, ball: Detection?) {
        let shape = output.shape
        let dim1 = shape.count > 1 ? shape[1] : 0
        let dim2 = shape.count > 2 ? shape[2] : 0

        let detections: [Detection]
        if dim2 == 6 {
            detections = decodeNMS(output, rows: dim1)
        } else {
            let classCount = min(dim1, dim2) - 4
            let anchors = max(dim1, dim2)
            detections = decodeRaw(output, classCount: classCount, anchors: anchors)
        }

        var person: Detection?
        var ball: Detection?
        for d in detections {
            if d.classIndex == AppConfig.personClassIndex,
               d.score > (person?.score ?? 0) { person = d }
            if d.classIndex == AppConfig.ballClassIndex,
               d.score > (ball?.score ?? 0) { ball = d }
        }
        return (person, ball)
    }

    private static func decodeNMS(_ output: Tensor, rows: Int) -> [Detection] {
        let data: [Float] = output.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let size = Float(AppConfig.detSize)
        var result: [Detection] = []
        for i in 0..<rows {
            let o = i * 6
            guard o + 5 < data.count else { break }
            let score = data[o + 4]
            guard score > AppConfig.detConfThreshold else { continue }
            let rect = CGRect(
                x: CGFloat(data[o] / size),
                y: CGFloat(data[o + 1] / size),
                width: CGFloat((data[o + 2] - data[o]) / size),
                height: CGFloat((data[o + 3] - data[o + 1]) / size)
            )
            result.append(Detection(rect: rect, score: score, classIndex: Int(data[o + 5])))
        }
        return result
    }

    private static func decodeRaw(_ output: Tensor, classCount: Int, anchors: Int) -> [Detection] {
        let data: [Float] = output.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let size = Float(AppConfig.detSize)
        // Only the two classes the demo uses — keeps the scan cheap.
        let wanted = [AppConfig.personClassIndex, AppConfig.ballClassIndex]
        var result: [Detection] = []
        for a in 0..<anchors {
            for c in wanted where c < classCount {
                let score = data[(4 + c) * anchors + a]
                guard score > AppConfig.detConfThreshold else { continue }
                let xc = data[a], yc = data[anchors + a]
                let w = data[2 * anchors + a], h = data[3 * anchors + a]
                let rect = CGRect(
                    x: CGFloat((xc - w / 2) / size),
                    y: CGFloat((yc - h / 2) / size),
                    width: CGFloat(w / size),
                    height: CGFloat(h / size)
                )
                result.append(Detection(rect: rect, score: score, classIndex: c))
            }
        }
        return nms(result)
    }

    private static func nms(_ boxes: [Detection]) -> [Detection] {
        let sorted = boxes.sorted { $0.score > $1.score }
        var kept: [Detection] = []
        for box in sorted {
            let overlaps = kept.contains {
                $0.classIndex == box.classIndex && iou($0.rect, box.rect) > CGFloat(AppConfig.iouThreshold)
            }
            if !overlaps { kept.append(box) }
        }
        return kept
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard inter.width > 0, inter.height > 0 else { return 0 }
        let interArea = inter.width * inter.height
        return interArea / (a.width * a.height + b.width * b.height - interArea)
    }
}
