import Foundation

/// The `cherrypad://` URL the keyboard uses to launch the container app. The full
/// payload lives in the App Group; the URL just carries the request id and acts as
/// the trigger to foreground the app.
enum DeepLink {
    static let scheme = "cherrypad"
    static let processHost = "process"

    /// Builds `cherrypad://process?id=<uuid>`.
    static func process(id: UUID) -> URL? {
        var components = URLComponents()
        components.scheme = scheme
        components.host = processHost
        components.queryItems = [URLQueryItem(name: "id", value: id.uuidString)]
        return components.url
    }

    /// Parses a `cherrypad://process?id=…` URL back to the request id.
    static func requestID(from url: URL) -> UUID? {
        guard url.scheme == scheme, url.host == processHost else { return nil }
        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        guard let value = components?.queryItems?.first(where: { $0.name == "id" })?.value else { return nil }
        return UUID(uuidString: value)
    }
}
