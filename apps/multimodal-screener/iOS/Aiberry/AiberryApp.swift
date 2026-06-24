import SwiftUI
import UIKit

@main
struct AiberryApp: App {
    @StateObject private var models = AppModels()

    init() {
        Self.configureAppearance()
        SpeechTranscriber.requestAuthorization()
    }

    var body: some Scene {
        WindowGroup {
            RootView(models: models)
                .task { models.preloadAll() }   // download both on-device models at launch
        }
    }

    /// Match the warm cream theme on the navigation/window chrome.
    private static func configureAppearance() {
        let cream = UIColor(red: 0.953, green: 0.945, blue: 0.918, alpha: 1)
        let nav = UINavigationBarAppearance()
        nav.configureWithOpaqueBackground()
        nav.backgroundColor = cream
        UINavigationBar.appearance().standardAppearance = nav
        UINavigationBar.appearance().scrollEdgeAppearance = nav
    }
}

/// Bridges the launch-owned `AppModels` into a `CheckInSession` (which needs the
/// voice + face models at init), then hands both to the flow view.
private struct RootView: View {
    @ObservedObject var models: AppModels
    @StateObject private var session: CheckInSession

    init(models: AppModels) {
        self.models = models
        _session = StateObject(wrappedValue: CheckInSession(voice: models.voice, face: models.face))
    }

    var body: some View {
        CheckInView(models: models, session: session)
    }
}
