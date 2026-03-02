import SwiftUI

struct DiagnosticsView: View {
    @EnvironmentObject var llmService: LLMService
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model Info")) {
                    HStack {
                        Text("Model ID")
                        Spacer()
                        Text("Qwen/Qwen3-4B").foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Personal Key")
                        Spacer()
                        Text("ztp_476b34ed6d484f03b19ef9ae74eddc4d").foregroundColor(.secondary)
                    }
                }
                
                Section(header: Text("Last Generation Metrics")) {
                    HStack {
                        Text("Time")
                        Spacer()
                        Text(String(format: "%.0f ms", llmService.lastGenerationTimeMs)).foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Token Count")
                        Spacer()
                        Text("\(llmService.lastTokenCount) tokens").foregroundColor(.secondary)
                    }
                    if llmService.lastGenerationTimeMs > 0 && llmService.lastTokenCount > 0 {
                        let speed = Double(llmService.lastTokenCount) / (llmService.lastGenerationTimeMs / 1000.0)
                        HStack {
                            Text("Speed")
                            Spacer()
                            Text(String(format: "%.2f tokens/s", speed)).foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Diagnostics")
        }
    }
}
