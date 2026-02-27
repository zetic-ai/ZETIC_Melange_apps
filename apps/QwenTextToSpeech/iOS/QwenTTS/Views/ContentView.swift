import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Qwen3-TTS CustomVoice")
                .font(.title)
                .fontWeight(.bold)
            
            Text(viewModel.status)
                .font(.subheadline)
                .foregroundColor(.gray)
            
            VStack(alignment: .leading) {
                Text("Input Text")
                    .font(.headline)
                TextEditor(text: $viewModel.inputText)
                    .frame(height: 100)
                    .padding(5)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
            .padding()
            
            HStack {
                Text("Speaker")
                Spacer()
                Picker("Speaker", selection: $viewModel.selectedSpeaker) {
                    ForEach(viewModel.speakers, id: \.self) { speaker in
                        Text(speaker).tag(speaker)
                    }
                }
                .pickerStyle(MenuPickerStyle())
            }
            .padding(.horizontal)
            
            Button(action: {
                viewModel.generate()
            }) {
                HStack {
                    if viewModel.isGenerating {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    }
                    Text(viewModel.isGenerating ? "Generating..." : "Generate Audio")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding()
            .disabled(viewModel.isGenerating || viewModel.status != "Ready")
            
            Spacer()
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
