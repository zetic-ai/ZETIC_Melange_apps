import SwiftUI

struct RootView: View {
    @StateObject private var viewModel = DemoViewModel()

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            switch viewModel.phase {
            case .loading:
                ModelDownloadView(viewModel: viewModel)
            case .missingClip:
                MissingClipView(viewModel: viewModel)
            case .running:
                DemoView(viewModel: viewModel)
            }
        }
    }
}
