//
//  ContentView.swift
//  NeuTTSNanoApp
//
//  Created by Assistant
//

import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var ttsManager = NeuTTSManager()

    @State private var inputText = "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all."
    @State private var referenceText = "Hello, this is a reference voice sample for voice cloning."
    @State private var showAudioPicker = false
    @State private var referenceAudioURL: URL?
    @State private var generatedAudioData: Data?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Status indicator
                    HStack {
                        Circle()
                            .fill(ttsManager.isInitialized ? Color.green : Color.red)
                            .frame(width: 12, height: 12)

                        Text(ttsManager.isInitialized ? "Models Ready" : ttsManager.statusMessage)
                            .font(.headline)

                        if ttsManager.isProcessing {
                            ProgressView()
                                .padding(.leading, 8)
                        }
                    }
                    .padding(.horizontal)

                    // Error message
                    if let error = ttsManager.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .padding()
                            .background(Color.red.opacity(0.1))
                            .cornerRadius(8)
                            .padding(.horizontal)
                    }

                    // Model loading logs
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Model Loading Logs")
                            .font(.headline)
                        ScrollView {
                            VStack(alignment: .leading, spacing: 4) {
                                ForEach(Array(ttsManager.logLines.enumerated()), id: \.offset) { _, line in
                                    Text(line)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 200)
                        .padding(8)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)

                        Button(action: {
                            Task {
                                await ttsManager.resetModelCache()
                            }
                        }) {
                            Text("Reset Model Cache (Redownload)")
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(.horizontal)

                    // Text input
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Text to Synthesize")
                            .font(.headline)

                        TextEditor(text: $inputText)
                            .frame(height: 100)
                            .padding(8)
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                            .disabled(!ttsManager.isInitialized)
                    }
                    .padding(.horizontal)

                    // Reference audio section
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Voice Cloning (Optional)")
                            .font(.headline)

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Reference Text")
                                .font(.subheadline)

                            TextEditor(text: $referenceText)
                                .frame(height: 60)
                                .padding(8)
                                .background(Color.gray.opacity(0.1))
                                .cornerRadius(8)
                                .disabled(!ttsManager.isInitialized)
                        }

                        Button(action: {
                            showAudioPicker = true
                        }) {
                            HStack {
                                Image(systemName: "waveform")
                                Text(referenceAudioURL != nil ? "Reference Audio Selected" : "Select Reference Audio")
                            }
                            .padding()
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(8)
                        }
                        .disabled(!ttsManager.isInitialized)

                        if let url = referenceAudioURL {
                            Text("Selected: \(url.lastPathComponent)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.horizontal)

                    // Action buttons
                    VStack(spacing: 12) {
                        Button(action: {
                            Task {
                                await synthesizeSpeech()
                            }
                        }) {
                            HStack {
                                Image(systemName: "play.circle.fill")
                                Text("Generate Speech")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(ttsManager.isInitialized && !ttsManager.isProcessing ? Color.blue : Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                        .disabled(!ttsManager.isInitialized || ttsManager.isProcessing)

                        if generatedAudioData != nil {
                            Button(action: {
                                if let data = generatedAudioData {
                                    ttsManager.playAudio(data: data)
                                }
                            }) {
                                HStack {
                                    Image(systemName: "speaker.wave.2.fill")
                                    Text("Play Generated Audio")
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                            }
                        }
                    }
                    .padding(.horizontal)

                    // Info section
                    VStack(alignment: .leading, spacing: 8) {
                        Text("About NeuTTS Nano")
                            .font(.headline)

                        Text("State-of-the-art Voice AI for on-device TTS with instant voice cloning. Ultra-fast generation on mobile CPUs with high realism and naturalness.")
                            .font(.body)
                            .foregroundColor(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.horizontal)
                    .padding(.top, 20)
                }
                .padding(.vertical)
            }
            .navigationTitle("NeuTTS Nano")
            .sheet(isPresented: $showAudioPicker) {
                AudioPickerView(selectedURL: $referenceAudioURL)
            }
        }
    }

    private func synthesizeSpeech() async {
        do {
            let referenceAudioData = try await loadReferenceAudioData()
            let audioData = try await ttsManager.synthesizeSpeech(
                text: inputText,
                referenceAudioData: referenceAudioData,
                referenceText: referenceText
            )

            await MainActor.run {
                generatedAudioData = audioData
            }
        } catch {
            await MainActor.run {
                ttsManager.errorMessage = "Synthesis failed: \(error.localizedDescription)"
            }
        }
    }

    private func loadReferenceAudioData() async throws -> Data? {
        guard let url = referenceAudioURL else { return nil }

        guard url.startAccessingSecurityScopedResource() else {
            throw NSError(domain: "Audio", code: -2, userInfo: [NSLocalizedDescriptionKey: "Permission denied for selected audio file"])
        }
        defer { url.stopAccessingSecurityScopedResource() }

        // Load audio file data
        let data = try Data(contentsOf: url)

        // Basic validation - check if it's a WAV file
        guard data.count > 44 else { // WAV header is at least 44 bytes
            throw NSError(domain: "Audio", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid audio file"])
        }

        return data
    }
}

struct AudioPickerView: UIViewControllerRepresentable {
    @Binding var selectedURL: URL?
    @Environment(\.presentationMode) var presentationMode

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.audio])
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let parent: AudioPickerView

        init(_ parent: AudioPickerView) {
            self.parent = parent
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            if let url = urls.first {
                parent.selectedURL = url
            }
            parent.presentationMode.wrappedValue.dismiss()
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
