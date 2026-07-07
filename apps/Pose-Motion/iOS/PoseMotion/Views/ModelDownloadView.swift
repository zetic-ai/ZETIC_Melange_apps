import SwiftUI

/// One row per Melange model while they download & compile on-device.
struct ModelDownloadView: View {
    @ObservedObject var viewModel: DemoViewModel

    var body: some View {
        VStack(spacing: 28) {
            VStack(spacing: 6) {
                Image(systemName: "figure.golf")
                    .font(.system(size: 44, weight: .semibold))
                    .foregroundStyle(Theme.accent)
                Text("Pose & Motion")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundStyle(Theme.textPrimary)
                Text("Preparing on-device models")
                    .font(.system(size: 14))
                    .foregroundStyle(Theme.textSecondary)
            }

            VStack(spacing: 12) {
                ForEach(Array(viewModel.loadStates.enumerated()), id: \.element.id) { index, state in
                    ModelLoadRow(state: state) {
                        viewModel.retry(index: index)
                    }
                }
            }
            .padding(.horizontal, 28)
        }
    }
}

private struct ModelLoadRow: View {
    let state: ModelLoadState
    let retry: () -> Void

    var body: some View {
        Card {
            HStack(spacing: 14) {
                ZStack {
                    Circle()
                        .stroke(Theme.accentSoft, lineWidth: 4)
                        .frame(width: 34, height: 34)
                    if state.loaded {
                        Image(systemName: "checkmark")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(Theme.good)
                    } else if state.error != nil {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 14))
                            .foregroundStyle(Theme.poor)
                    } else {
                        Circle()
                            .trim(from: 0, to: CGFloat(max(0.03, state.progress)))
                            .stroke(Theme.accent, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                            .frame(width: 34, height: 34)
                            .rotationEffect(.degrees(-90))
                            .animation(.easeOut(duration: 0.25), value: state.progress)
                    }
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(state.id)
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .foregroundStyle(Theme.textPrimary)
                    if let error = state.error {
                        Text(state.optional ? "Unavailable — demo continues in 2D" : error)
                            .font(.system(size: 12))
                            .foregroundStyle(state.optional ? Theme.textSecondary : Theme.poor)
                            .lineLimit(2)
                    } else if state.loaded {
                        Text("Ready")
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.textSecondary)
                    } else {
                        Text(state.progress > 0
                             ? "Downloading \(Int(state.progress * 100))%"
                             : "Optimizing for this device…")
                            .font(.system(size: 12))
                            .monospacedDigit()
                            .foregroundStyle(Theme.textSecondary)
                    }
                }
                Spacer()

                if state.error != nil {
                    Button(action: retry) {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(Theme.accent)
                    }
                }
            }
            .padding(14)
        }
    }
}
