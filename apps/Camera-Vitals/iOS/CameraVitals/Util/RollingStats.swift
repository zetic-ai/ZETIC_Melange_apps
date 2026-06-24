import Foundation

/// Median window followed by an EMA — keeps the displayed BPM steady so a single
/// noisy window never flashes a wild number on screen.
struct MedianEMA {
    private var window: [Double] = []
    private let size: Int
    private let alpha: Double
    private(set) var value: Double?

    init(size: Int = 5, alpha: Double = 0.3) {
        self.size = size
        self.alpha = alpha
    }

    @discardableResult
    mutating func update(_ x: Double) -> Double {
        window.append(x)
        if window.count > size { window.removeFirst() }
        let med = window.sorted()[window.count / 2]
        if let v = value {
            value = alpha * med + (1 - alpha) * v
        } else {
            value = med
        }
        return value!
    }

    mutating func reset() {
        window.removeAll()
        value = nil
    }
}
