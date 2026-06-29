import SwiftUI

/// Top-level router: download gate → capture → results, all over the aurora backdrop.
struct RootView: View {
    @EnvironmentObject private var vm: DiagnosisViewModel

    var body: some View {
        ZStack {
            Theme.background

            Group {
                if !vm.canAnalyze {
                    DownloadView()
                        .transition(.opacity)
                } else if vm.analysis == .none {
                    CaptureView()
                        .transition(.asymmetric(insertion: .opacity, removal: .opacity))
                } else {
                    ResultsView()
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .animation(.easeInOut(duration: 0.45), value: vm.canAnalyze)
            .animation(.easeInOut(duration: 0.4), value: vm.analysis)
        }
        .onAppear { vm.bootstrap() }
    }
}
