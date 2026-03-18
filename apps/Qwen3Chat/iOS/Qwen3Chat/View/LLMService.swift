import Foundation
import Combine
import ZeticMLange

@MainActor
class LLMService: ObservableObject {
    @Published var isGenerating = false
    @Published var currentStreamText = ""
    @Published var lastGenerationTimeMs: Double = 0
    @Published var lastTokenCount: Int = 0
    @Published var downloadProgress: Float = 0.0
    @Published var isDownloading = true
    @Published var initializationState: String = "Checking Model..."
    
    private let modelId = "Qwen/Qwen3-4B"
    // Hidden from logs, directly used
    private let personalKey = "YOUR_MLANGE_KEY"
    
    private var model: ZeticMLangeLLMModel?
    private var generationTask: Task<Void, Never>?
    
    init() {}
    
    func loadModel() {
        guard model == nil && isDownloading else { return }
        Task.detached { [weak self] in
            guard let self = self else { return }
            do {
                let loadedModel = try ZeticMLangeLLMModel(personalKey: self.personalKey, name: self.modelId) { progress in
                    Task { @MainActor in
                        self.downloadProgress = progress
                        if progress > 0.0 {
                            self.initializationState = "Downloading Model (\(Int(progress * 100))%)"
                        }
                    }
                }
                await MainActor.run {
                    self.model = loadedModel
                    self.isDownloading = false
                }
            } catch {
                print("Failed to initialize model: \(error)")
                await MainActor.run {
                    self.isDownloading = false
                }
            }
        }
    }
    
    func generateResponse(prompt: String, onResponseComplete: @escaping (String) -> Void) {
        guard let model = model else { return }
        
        isGenerating = true
        currentStreamText = ""
        lastTokenCount = 0
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        generationTask = Task.detached { [weak self] in
            guard let self = self else { return }
            
            do {
                try model.cleanUp()
            } catch {
                print("Cleanup failed: \(error)")
            }
            
            do {
                try model.run(prompt)
                
                var totalTokens = 0
                while !Task.isCancelled {
                    let token = model.waitForNextToken().token
                    if token.isEmpty { break }
                    
                    totalTokens += 1
                    
                    await MainActor.run {
                        self.currentStreamText += token
                    }
                }
                
                let duration = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
                
                await MainActor.run {
                    self.lastTokenCount = totalTokens
                    self.lastGenerationTimeMs = duration
                    onResponseComplete(self.currentStreamText)
                    self.isGenerating = false
                    self.currentStreamText = ""
                }
                
                do {
                    try model.cleanUp()
                } catch {
                    print("Cleanup failed: \(error)")
                }
                
            } catch {
                await MainActor.run {
                    self.isGenerating = false
                    self.currentStreamText = "Error during generation."
                }
            }
        }
    }
    
    func stopGeneration() {
        generationTask?.cancel()
        do {
            try model?.cleanUp()
        } catch {
            print("Cleanup failed: \(error)")
        }
        isGenerating = false
    }
    
    func cleanUp() {
        do {
            try model?.cleanUp()
        } catch {
            print("Cleanup failed: \(error)")
        }
    }
}
