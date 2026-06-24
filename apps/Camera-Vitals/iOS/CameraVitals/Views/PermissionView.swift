import SwiftUI
import UIKit

/// Shown when camera access is denied.
struct PermissionView: View {
    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "camera.metering.unknown")
                .font(.system(size: 44))
                .foregroundStyle(Theme.accent)
            Text("Camera access needed")
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(Theme.textPrimary)
            Text("Camera-Vitals reads your pulse from the front camera. Video never leaves your phone.")
                .font(.system(size: 15))
                .foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            Button {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            } label: {
                Text("Open Settings")
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 28).padding(.vertical, 12)
                    .background(Theme.accent, in: Capsule())
            }
        }
        .padding()
    }
}
