import SwiftUI
import Network

/// "Nothing leaves this phone" — the privacy pitch, styled as a soft card with a
/// live network indicator so you can prove it still works in Airplane Mode.
struct PrivacyBanner: View {
    @StateObject private var net = NetworkMonitor()

    var body: some View {
        HStack(spacing: 12) {
            IconTile(system: "lock.shield.fill", size: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text("100% on-device").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.ink)
                Text("Voice & video never leave this phone").font(.caption).foregroundStyle(Theme.inkSoft)
            }
            Spacer()
            Label(net.isOnline ? "Online" : "Offline",
                  systemImage: net.isOnline ? "wifi" : "airplane")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(net.isOnline ? Theme.inkSoft : Theme.tileInk)
        }
        .card(14)
    }
}

/// On-device inference latency chip.
struct LatencyBadge: View {
    let ms: Double?
    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "bolt.fill").foregroundStyle(Theme.warn)
            if let ms {
                Text("\(ms, specifier: "%.0f") ms").font(.caption.bold().monospacedDigit())
                    .foregroundStyle(Theme.ink)
                Text("on-device").font(.caption2).foregroundStyle(Theme.inkSoft)
            } else {
                Text("—").font(.caption.bold()).foregroundStyle(Theme.inkSoft)
            }
        }
        .padding(.horizontal, 11).padding(.vertical, 6)
        .background(Capsule().fill(Theme.bg))
    }
}

/// Lightweight reachability used only to display online/offline state.
final class NetworkMonitor: ObservableObject {
    @Published var isOnline = true
    private let monitor = NWPathMonitor()
    init() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async { self?.isOnline = path.status == .satisfied }
        }
        monitor.start(queue: DispatchQueue(label: "aiberry.net"))
    }
    deinit { monitor.cancel() }
}
