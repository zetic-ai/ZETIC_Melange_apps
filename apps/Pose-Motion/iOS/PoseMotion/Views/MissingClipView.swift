import SwiftUI

/// Shown when no bundled clip is found; documents the drop-in path.
struct MissingClipView: View {
    @ObservedObject var viewModel: DemoViewModel

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "video.slash")
                .font(.system(size: 40, weight: .semibold))
                .foregroundStyle(Theme.textSecondary)
            Text("No demo clip found")
                .font(.system(size: 21, weight: .bold, design: .rounded))
                .foregroundStyle(Theme.textPrimary)
            VStack(alignment: .leading, spacing: 10) {
                Label("Add `GolfSwing.mp4` to `PoseMotion/Media/` and rebuild", systemImage: "1.circle")
                Label("Or copy it into this app's Documents folder via Finder / the Files app", systemImage: "2.circle")
                Label("Side-on view, single athlete, visible ball, 720–1080p works best", systemImage: "lightbulb")
            }
            .font(.system(size: 14))
            .foregroundStyle(Theme.textSecondary)
            .padding(.horizontal, 32)

            Button {
                viewModel.recheckClip()
            } label: {
                Text("Check again")
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .padding(.horizontal, 22)
                    .padding(.vertical, 10)
                    .background(Theme.accent, in: Capsule())
                    .foregroundStyle(.black)
            }
        }
    }
}
