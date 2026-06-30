import SwiftUI

/// App root: warms the model at launch, hosts the compose screen, and routes
/// keyboard handoffs (deep link + foreground fallback) into the AppModel.
struct RootView: View {
    @StateObject private var model = AppModel()
    @ObservedObject private var llm = LLMService.shared
    @Environment(\.scenePhase) private var scenePhase

    @AppStorage("hasSeenOnboarding") private var hasSeenOnboarding = false
    @State private var showSettings = false
    @State private var showOnboarding = false

    var body: some View {
        ComposeScreen(model: model, llm: llm, showSettings: $showSettings)
            .task {
                // Warm the model once at launch so a tapped action has zero load cost.
                await llm.ensureLoaded()
            }
            .onAppear {
                if !hasSeenOnboarding { showOnboarding = true }
            }
            .onOpenURL { url in
                model.handleDeepLink(url)
            }
            .onChange(of: scenePhase) { _, phase in
                // Fallback in case the launch URL was dropped: pick up a pending
                // keyboard request when we become active.
                if phase == .active { model.consumePendingRequest() }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(llm: llm, onShowOnboarding: {
                    showSettings = false
                    showOnboarding = true
                })
            }
            .sheet(isPresented: $showOnboarding) {
                OnboardingView(onDone: {
                    showOnboarding = false
                    hasSeenOnboarding = true
                })
            }
    }
}
