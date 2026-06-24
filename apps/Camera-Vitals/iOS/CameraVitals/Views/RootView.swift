import SwiftUI

/// Routes between screens based on the FSM, and overlays the report sheet.
struct RootView: View {
    @StateObject private var vm = VitalsViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            switch vm.state {
            case .loadingModel(let progress):
                DownloadView(progress: progress)
            case .permissionDenied:
                PermissionView()
            case .error(let message):
                ErrorView(message: message) { vm.retry() }
            case .warmup, .live:
                MeasureView(vm: vm)
            }
        }
        .sheet(item: $vm.report) { report in
            ReportView(report: report) { vm.dismissReport() }
        }
        .onAppear { vm.start() }
        .onChange(of: scenePhase) { phase in
            switch phase {
            case .active: if vm.isReady { vm.camera.start() }
            case .background: vm.stop()
            default: break
            }
        }
    }
}

/// Simple error screen with retry.
struct ErrorView: View {
    let message: String
    let onRetry: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 44))
                .foregroundStyle(Theme.fair)
            Text("Something went wrong")
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(Theme.textPrimary)
            Text(message)
                .font(.system(size: 15))
                .foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            Button(action: onRetry) {
                Text("Try again")
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 28).padding(.vertical, 12)
                    .background(Theme.accent, in: Capsule())
            }
        }
        .padding()
    }
}
