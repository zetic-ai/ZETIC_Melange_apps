import SwiftUI

struct ChatView: View {
    @EnvironmentObject var llmService: LLMService
    @EnvironmentObject var sessionManager: ChatSessionManager
    
    @State private var inputText: String = ""
    
    var body: some View {
        NavigationView {
            VStack {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            if llmService.isDownloading {
                                VStack(spacing: 8) {
                                    Text(llmService.initializationState)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    
                                    if llmService.downloadProgress > 0.0 && llmService.downloadProgress < 1.0 {
                                        ProgressView(value: llmService.downloadProgress)
                                            .progressViewStyle(LinearProgressViewStyle(tint: Color(hex: "#34A9A3")))
                                    } else {
                                        ProgressView()
                                            .progressViewStyle(CircularProgressViewStyle())
                                    }
                                }
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(12)
                                .padding(.vertical, 8)
                            }
                            
                            ForEach(sessionManager.messages, id: \.id) { msg in
                                ChatBubble(text: msg.text, isUser: msg.isUser)
                                    .id(msg.id)
                            }
                            
                            if llmService.isGenerating && !llmService.currentStreamText.isEmpty {
                                ChatBubble(text: llmService.currentStreamText, isUser: false)
                                    .id("generating")
                            } else if llmService.isGenerating {
                                HStack {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle())
                                    Text("Loading Model...")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .padding()
                                .id("loading")
                            }
                        }
                        .padding()
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .onTapGesture {
                        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                    }
                    .onChange(of: sessionManager.messages.count) { _ in
                        scrollToBottom(proxy: proxy)
                    }
                    .onChange(of: llmService.currentStreamText) { _ in
                        scrollToBottom(proxy: proxy, id: "generating")
                    }
                }
                
                HStack(alignment: .bottom) {
                    TextField("Message Qwen3-4B...", text: $inputText, axis: .vertical)
                        .lineLimit(1...5)
                        .padding(10)
                        .background(Color(.systemGray6))
                        .cornerRadius(20)
                    
                    if llmService.isGenerating {
                        Button(action: {
                            llmService.stopGeneration()
                            sessionManager.addMessage(text: llmService.currentStreamText, isUser: false)
                            llmService.currentStreamText = ""
                        }) {
                            Image(systemName: "stop.circle.fill")
                                .font(.system(size: 30))
                                .foregroundColor(.red)
                        }
                    } else {
                        Button(action: sendMessage) {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.system(size: 30))
                                .foregroundColor(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .gray : Color(hex: "#34A9A3"))
                        }
                        .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
                .padding()
            }
            .navigationTitle("Qwen3 Chat")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                llmService.loadModel { errorMsg in
                    sessionManager.addMessage(text: errorMsg, isUser: false)
                }
            }
        }
    }
    
    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        
        sessionManager.addMessage(text: text, isUser: true)
        inputText = ""
        
        let prompt = sessionManager.buildPrompt()
        llmService.generateResponse(prompt: prompt) { fullResponse in
            if !fullResponse.isEmpty {
                sessionManager.addMessage(text: fullResponse, isUser: false)
            }
        }
    }
    
    private func scrollToBottom(proxy: ScrollViewProxy, id: String? = nil) {
        if let id = id {
            withAnimation { proxy.scrollTo(id, anchor: .bottom) }
        } else if let last = sessionManager.messages.last {
            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
        }
    }
}

struct ChatBubble: View {
    let text: String
    let isUser: Bool
    
    var body: some View {
        HStack {
            if isUser { Spacer() }
            
            Text(text)
                .padding(12)
                .background(isUser ? Color(hex: "#34A9A3") : Color(.systemGray5))
                .foregroundColor(isUser ? .white : .primary)
                .cornerRadius(16)
                .clipShape(ChatBubbleShape(isUser: isUser))
            
            if !isUser { Spacer() }
        }
    }
}

struct ChatBubbleShape: Shape {
    let isUser: Bool
    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: [.topLeft, .topRight, isUser ? .bottomLeft : .bottomRight],
            cornerRadii: CGSize(width: 16, height: 16)
        )
        return Path(path.cgPath)
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default: (a, r, g, b) = (1, 1, 1, 0)
        }
        self.init(.sRGB, red: Double(r) / 255, green: Double(g) / 255, blue:  Double(b) / 255, opacity: Double(a) / 255)
    }
}
