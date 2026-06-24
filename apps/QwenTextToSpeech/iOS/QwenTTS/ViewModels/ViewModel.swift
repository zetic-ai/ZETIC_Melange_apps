import Foundation
import Combine

class ViewModel: ObservableObject {
    @Published var inputText: String = "你好，我是Qwen Text to Speech。"
    @Published var selectedSpeaker: String = "Vivian"
    @Published var status: String = "Initializing..."
    @Published var isGenerating: Bool = false
    
    let speakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
    
    private let pipeline = QwenPipeline()
    private let audioManager = AudioManager()
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        pipeline.$status
            .receive(on: DispatchQueue.main)
            .assign(to: \.status, on: self)
            .store(in: &cancellables)
            
        Task {
            await pipeline.load()
        }
    }
    
    func generate() {
        guard !isGenerating else { return }
        isGenerating = true
        
        Task {
            do {
                let wav = try await pipeline.generate(text: inputText, speaker: selectedSpeaker)
                if !wav.isEmpty {
                    DispatchQueue.main.async {
                        self.audioManager.play(pcmData: wav)
                    }
                }
            } catch {
                print("Generation error: \(error)")
            }
            
            DispatchQueue.main.async {
                self.isGenerating = false
            }
        }
    }
}
