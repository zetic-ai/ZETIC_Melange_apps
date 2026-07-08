import SwiftUI

/// Video with live overlays, the benchmark HUD, the 3D panel, and mode controls.
struct DemoView: View {
    @ObservedObject var viewModel: DemoViewModel

    var body: some View {
        GeometryReader { geo in
            let fit = VideoFitMapper.fitRect(content: viewModel.frameSize, in: geo.size)

            ZStack {
                if let image = viewModel.frameImage {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .frame(width: fit.width, height: fit.height)
                        .position(x: fit.midX, y: fit.midY)
                } else {
                    ProgressView()
                        .tint(Theme.accent)
                }

                BoundingBoxOverlay(personBox: viewModel.personBox,
                                   ballBox: viewModel.ballBox, fit: fit)
                BallTrailOverlay(trail: viewModel.ballTrail, fit: fit)
                if let keypoints = viewModel.keypoints {
                    SkeletonOverlay(keypoints: keypoints, fit: fit)
                }

                VStack {
                    HStack(alignment: .top) {
                        BenchmarkHUD(stats: viewModel.stats,
                                     liftAvailable: viewModel.liftAvailable,
                                     mode: viewModel.mode)
                        Spacer()
                        if viewModel.availableClips.count > 1 {
                            clipPicker
                        }
                    }
                    Spacer()
                    HStack(alignment: .bottom) {
                        modePicker
                        Spacer()
                        if viewModel.liftAvailable {
                            pose3DPanel
                        }
                    }
                }
                .padding(12)
            }
        }
        .background(Theme.background)
        .statusBarHidden()
    }

    /// One numbered chip per bundled clip (shown only when there is a choice).
    private var clipPicker: some View {
        VStack(spacing: 4) {
            ForEach(Array(viewModel.availableClips.enumerated()), id: \.element) { index, name in
                let active = name == viewModel.selectedClip
                Button {
                    viewModel.setClip(name)
                } label: {
                    Text("\(index + 1)")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                        .frame(width: 26, height: 26)
                        .background(active ? Theme.accent : .clear, in: Circle())
                        .foregroundStyle(active ? .black : Theme.textSecondary)
                }
            }
        }
        .padding(3)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var modePicker: some View {
        HStack(spacing: 0) {
            ForEach(ClipFrameSource.Mode.allCases) { mode in
                Button {
                    viewModel.setMode(mode)
                } label: {
                    Text(mode.rawValue)
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(viewModel.mode == mode ? Theme.accent : .clear, in: Capsule())
                        .foregroundStyle(viewModel.mode == mode ? .black : Theme.textSecondary)
                }
            }
        }
        .padding(3)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var pose3DPanel: some View {
        VStack(spacing: 4) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { viewModel.show3D.toggle() }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: viewModel.show3D ? "cube.fill" : "cube")
                        .font(.system(size: 11, weight: .semibold))
                    Text("3D")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                .foregroundStyle(viewModel.show3D ? Theme.accent : Theme.textSecondary)
            }

            if viewModel.show3D {
                Group {
                    if let joints = viewModel.pose3D {
                        Skeleton3DCanvas(joints: joints)
                    } else {
                        Text("Gathering motion…")
                            .font(.system(size: 11, design: .rounded))
                            .foregroundStyle(Theme.textSecondary)
                    }
                }
                .frame(width: 150, height: 170)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
        }
    }
}
