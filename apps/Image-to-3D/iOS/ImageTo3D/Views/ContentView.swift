import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ImageTo3DViewModel()
    @State private var pickerSource: UIImagePickerController.SourceType?

    var body: some View {
        VStack(spacing: 12) {
            header

            switch viewModel.phase {
            case .loadingModel(let progress):
                loadingModelView(progress)
            case .idle:
                emptyStateView
            case .processing(let stage):
                processingView(stage)
            case .ready:
                resultView
            case .error(let message):
                errorView(message)
            }

            Spacer(minLength: 0)
            controls
        }
        .padding()
        .background(Color(white: 0.03).ignoresSafeArea())
        .onAppear { viewModel.loadModel() }
        .sheet(item: $pickerSource) { source in
            ImagePicker(sourceType: source) { image in
                viewModel.process(image)
            }
        }
    }

    private var header: some View {
        VStack(spacing: 2) {
            Text("Image to 3D")
                .font(.title2.bold())
            Text("On-device depth analysis · ZETIC.MLange")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private func loadingModelView(_ progress: Float) -> some View {
        VStack(spacing: 12) {
            Spacer()
            ProgressView(value: progress > 0 ? progress : nil)
                .progressViewStyle(.linear)
                .frame(maxWidth: 240)
            Text(progress > 0
                 ? "Downloading model… \(Int(progress * 100))%"
                 : "Preparing model…")
                .font(.footnote)
                .foregroundColor(.secondary)
            Spacer()
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "cube.transparent")
                .font(.system(size: 56))
                .foregroundColor(.secondary)
            Text("Pick or take a photo to analyze its depth in 3D — fully offline.")
                .font(.footnote)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 260)
            Spacer()
        }
    }

    private func processingView(_ stage: String) -> some View {
        VStack(spacing: 12) {
            Spacer()
            ProgressView()
            Text(stage)
                .font(.footnote)
                .foregroundColor(.secondary)
            Spacer()
        }
    }

    private var resultView: some View {
        VStack(spacing: 12) {
            // Top: compact interactive 3D relief with mode toggle overlaid.
            ZStack(alignment: .bottom) {
                if let mesh = viewModel.mesh, let texture = viewModel.texture {
                    Model3DView(mesh: mesh,
                                texture: texture,
                                contentID: viewModel.contentID,
                                mode: viewModel.renderMode)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                Picker("Mode", selection: $viewModel.renderMode) {
                    ForEach(RenderMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 180)
                .padding(.bottom, 8)
            }
            .frame(height: 230)

            // Main display: photo → depth analysis, side by side.
            ZStack(alignment: .topTrailing) {
                HStack(spacing: 12) {
                    pane(viewModel.photo, label: "Photo")
                    pane(viewModel.depthImage, label: "Depth")
                }
                LatencyHUDView(latency: viewModel.latency)
                    .padding(6)
            }
            .frame(maxHeight: .infinity)
        }
    }

    private func errorView(_ message: String) -> some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 40))
                .foregroundColor(.orange)
            Text(message)
                .font(.footnote)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 280)
            Button("Retry model load") {
                viewModel.phase = .loadingModel(0)
                viewModel.loadModel()
            }
            .buttonStyle(.bordered)
            Spacer()
        }
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button {
                pickerSource = .photoLibrary
            } label: {
                Label("Photo Library", systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            Button {
                pickerSource = .camera
            } label: {
                Label("Camera", systemImage: "camera")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))
        }
        .disabled(!isInteractive)
    }

    private var isInteractive: Bool {
        switch viewModel.phase {
        case .idle, .ready, .error: return true
        case .loadingModel, .processing: return false
        }
    }
}

extension UIImagePickerController.SourceType: Identifiable {
    public var id: Int { rawValue }
}

private extension ContentView {
    func pane(_ image: UIImage?, label: String) -> some View {
        VStack(spacing: 6) {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color(white: 0.12))
            }
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}
