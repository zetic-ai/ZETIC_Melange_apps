import Foundation
import CoreMedia

class CameraFrame {
    private let imageBuffer: Any?
    private(set) var imageAddress: UnsafeMutableRawPointer
    let width: Int32
    let height: Int32
    let bytesPerRow: Int32
    
    init(buffer: Any, address: UnsafeMutableRawPointer, width: Int32, height: Int32, bytesPerRow: Int32) {
        self.imageBuffer = buffer
        self.imageAddress = address
        self.width = width
        self.height = height
        self.bytesPerRow = bytesPerRow
    }
    
    convenience init?(from sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)!
        let width = Int32(CVPixelBufferGetWidth(pixelBuffer))
        let height = Int32(CVPixelBufferGetHeight(pixelBuffer))
        let bytesPerRow = Int32(CVPixelBufferGetBytesPerRow(pixelBuffer))
        
        self.init(buffer: pixelBuffer, address: baseAddress, width: width, height: height, bytesPerRow: bytesPerRow)
    }
    
    func withUnsafeBytes<T>(_ body: (UnsafeMutableRawBufferPointer) throws -> T) rethrows -> T {
        let bufferSize = Int(height * bytesPerRow)
        let buffer = UnsafeMutableRawBufferPointer(start: imageAddress, count: bufferSize)
        return try body(buffer)
    }
}
