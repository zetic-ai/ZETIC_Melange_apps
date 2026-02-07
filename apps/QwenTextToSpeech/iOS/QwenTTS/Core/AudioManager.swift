import Foundation
import AVFoundation

class AudioManager: NSObject, ObservableObject {
    private var engine: AVAudioEngine!
    private var player: AVAudioPlayerNode!
    
    override init() {
        super.init()
        setupAudio()
    }
    
    private func setupAudio() {
        engine = AVAudioEngine()
        player = AVAudioPlayerNode()
        engine.attach(player)
        
        let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1) // Qwen 12Hz usually 24k? Verify rate
        engine.connect(player, to: engine.mainMixerNode, format: format)
        
        do {
            try engine.start()
        } catch {
            print("Audio Engine Start Error: \(error)")
        }
    }
    
    func play(pcmData: [Float], sampleRate: Double = 24000) {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(pcmData.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(pcmData.count)
        
        if let channelData = buffer.floatChannelData {
            for i in 0..<pcmData.count {
                channelData[0][i] = pcmData[i]
            }
        }
        
        player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
        if !player.isPlaying {
            player.play()
        }
    }
}
