import Foundation
import ZeticMLange

class TencentHYMT: ObservableObject {
    @Published var isLoaded: Bool = false
    @Published var isLoading: Bool = false
    @Published var loadingStatus: String = "Initializing Model..."
    @Published var error: String? = nil
    @Published var currentOutput: String = ""
    @Published var isGenerating: Bool = false
    
    private var model: ZeticMLangeLLMModel?
    private let projectKey = "YOUR_MLANGE_KEY"
    private let modelId = "vaibhav-zetic/tencent_HY-MT"
    
    private var currentSource: Language?
    private var currentTarget: Language?

    func loadModel() {
        guard !isLoaded && !isLoading else { return }
        
        isLoading = true
        loadingStatus = "Downloading Model..."
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            do {
                self.model = try ZeticMLangeLLMModel(tokenKey: self.projectKey, name: self.modelId)
                
                DispatchQueue.main.async {
                    self.isLoaded = true
                    self.isLoading = false
                    self.loadingStatus = "Model Loaded"
                }
            } catch {
                DispatchQueue.main.async {
                    self.error = "Error initializing: \(error.localizedDescription)"
                    self.isLoading = false
                    self.loadingStatus = "Error"
                }
            }
        }
    }
    
    func run(prompt: String, source: Language, target: Language) {
        // Ensure model is loaded or reload if context changed
        
        isGenerating = true
        currentOutput = ""
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            do {
                let isSwap = (source == self.currentTarget && target == self.currentSource)
                let isChange = (self.currentSource != source || self.currentTarget != target)
                
                if self.model == nil {
                    // Initial load if nil
                    self.model = try ZeticMLangeLLMModel(tokenKey: self.projectKey, name: self.modelId)
                    self.loadingStatus = "Model Loaded"
                } else if isChange {
                     DispatchQueue.main.async {
                        self.loadingStatus = "Cleaning Context..."
                    }
                    // Always clean up context when language changes
                    try self.model?.cleanUp()
                }
                
                // Update references
                if isChange {
                    self.currentSource = source
                    self.currentTarget = target
                }
                
                guard let model = self.model else { return }
                
                try model.run(prompt)
                
                while true {
                    let token = model.waitForNextToken()
                    
                    if token == "" {
                        break
                    }
                    
                    DispatchQueue.main.async {
                        self.currentOutput += token
                    }
                }
                
                try? model.cleanUp()
                
                DispatchQueue.main.async {
                    self.isGenerating = false
                    self.loadingStatus = "Model Loaded"
                }
                
            } catch {
                DispatchQueue.main.async {
                    self.error = "Error running model: \(error.localizedDescription)"
                    self.isGenerating = false
                    self.loadingStatus = "Error"
                }
            }
        }
    }
    
    deinit {
        try? model?.cleanUp()
    }
}
