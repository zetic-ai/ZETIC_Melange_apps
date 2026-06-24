import SwiftUI

/// Big serif headline with a semibold emphasis word (no italic).
struct EditorialTitle: View {
    let leading: String
    let emphasis: String
    var body: some View {
        (Text(leading + " ").font(.serif(40))
         + Text(emphasis).font(.serif(40, .semibold)))
            .foregroundStyle(Theme.ink)
            .lineLimit(2)
            .minimumScaleFactor(0.7)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Rounded sage tile holding a line-art glyph (the reference's note/voice icons).
struct IconTile: View {
    let system: String
    var size: CGFloat = 56
    var body: some View {
        RoundedRectangle(cornerRadius: 15, style: .continuous)
            .fill(Theme.tile)
            .frame(width: size, height: size)
            .overlay(Image(systemName: system)
                .font(.system(size: size * 0.42, weight: .regular))
                .foregroundStyle(Theme.tileInk))
    }
}

/// Section header inside a card: sage tile + title + caption.
struct CardHeader: View {
    let icon: String
    let title: String
    let subtitle: String
    var body: some View {
        HStack(spacing: 14) {
            IconTile(system: icon, size: 52)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.headline).foregroundStyle(Theme.ink)
                Text(subtitle).font(.caption).foregroundStyle(Theme.inkSoft)
            }
            Spacer(minLength: 0)
        }
    }
}

/// Large tap-to-record button with a live input-level ring.
struct RecordButton: View {
    let isRecording: Bool
    let level: Float
    let busy: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(isRecording ? Theme.dark : Theme.tile)
                    .frame(width: 128, height: 128)
                Circle()
                    .stroke(isRecording ? Theme.danger : Theme.tileInk.opacity(0.35),
                            lineWidth: 2 + CGFloat(level) * 12)
                    .frame(width: 128, height: 128)
                    .animation(.easeOut(duration: 0.08), value: level)
                if busy {
                    ProgressView().tint(Theme.tileInk).scaleEffect(1.3)
                } else {
                    Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                        .font(.system(size: 40, weight: .semibold))
                        .foregroundStyle(isRecording ? .white : Theme.tileInk)
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(busy)
    }
}

/// Horizontal probability bar with a label and percentage.
struct BarRow: View {
    let label: String
    let value: Float
    var tint: Color = Theme.accent
    var highlighted: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(label).font(.subheadline.weight(highlighted ? .semibold : .regular))
                    .foregroundStyle(Theme.ink)
                Spacer()
                Text("\(Int((value * 100).rounded()))%")
                    .font(.subheadline.monospacedDigit()).foregroundStyle(Theme.inkSoft)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.bg)
                    Capsule().fill(tint.opacity(highlighted ? 1 : 0.7))
                        .frame(width: max(6, geo.size.width * CGFloat(min(max(value, 0), 1))))
                        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: value)
                }
            }
            .frame(height: 10)
        }
    }
}

/// Status line describing what the on-device model is doing.
struct StatusLine: View {
    let status: ModelStatus
    var body: some View {
        Group {
            switch status {
            case .idle:                Label("Ready", systemImage: "circle")
            case .downloading(let p):  Label("Downloading model… \(Int(p*100))%", systemImage: "arrow.down.circle")
            case .loading:             Label("Preparing on NPU…", systemImage: "cpu")
            case .running:             Label("Analyzing on-device…", systemImage: "waveform")
            case .ready:               Label("Done", systemImage: "checkmark.circle.fill")
            case .failed(let m):       Label(m, systemImage: "exclamationmark.triangle.fill")
            }
        }
        .font(.caption)
        .foregroundStyle(status.isFailure ? Theme.danger : Theme.inkSoft)
        .lineLimit(2)
    }
}

private extension ModelStatus {
    var isFailure: Bool { if case .failed = self { return true } else { return false } }
}
