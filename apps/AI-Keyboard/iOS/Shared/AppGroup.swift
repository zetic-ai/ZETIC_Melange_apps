import Foundation

/// Single source of truth for the App Group shared between the container app and
/// the keyboard extension. The id MUST be byte-identical to the string in both
/// `.entitlements` files and must be registered on the Apple Developer portal for
/// the signing team, or `UserDefaults(suiteName:)` silently returns the standard
/// domain and cross-process writes vanish.
enum AppGroup {
    static let id = "group.ai.zetic.demo.cherrypad"

    /// Shared defaults. Falls back to `.standard` only if the suite can't be
    /// opened (e.g. the Simulator preview target, which isn't provisioned for the
    /// group) so UI iteration never crashes; on a correctly-provisioned device the
    /// suite is always available.
    static var defaults: UserDefaults {
        UserDefaults(suiteName: id) ?? .standard
    }

    /// Shared on-disk container, for payloads too large for `UserDefaults`.
    static var container: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: id)
    }
}
