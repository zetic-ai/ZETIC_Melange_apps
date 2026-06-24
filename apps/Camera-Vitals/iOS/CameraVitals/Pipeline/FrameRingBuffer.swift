import Foundation

/// Fixed-capacity ring of preprocessed frames (planar RGB float, CHW per frame).
/// Thread-safe: capture thread appends, inference thread snapshots.
final class FrameRingBuffer {
    private let capacity: Int
    private let frameLen: Int
    private var storage: [[Float]]
    private var head = 0          // next write slot; when full this is also the oldest
    private var count = 0
    private let lock = NSLock()

    init(capacity: Int, frameLen: Int) {
        self.capacity = capacity
        self.frameLen = frameLen
        self.storage = Array(repeating: [Float](repeating: 0, count: frameLen), count: capacity)
    }

    var isFull: Bool {
        lock.lock(); defer { lock.unlock() }
        return count >= capacity
    }

    var filled: Int {
        lock.lock(); defer { lock.unlock() }
        return count
    }

    func append(_ frame: [Float]) {
        lock.lock(); defer { lock.unlock() }
        storage[head] = frame
        head = (head + 1) % capacity
        if count < capacity { count += 1 }
    }

    func reset() {
        lock.lock(); defer { lock.unlock() }
        count = 0
        head = 0
    }

    /// Latest `capacity` frames in temporal order (oldest → newest), flattened.
    /// Returns nil until the buffer has filled.
    func snapshot() -> [Float]? {
        lock.lock(); defer { lock.unlock() }
        guard count >= capacity else { return nil }
        var out = [Float]()
        out.reserveCapacity(capacity * frameLen)
        for i in 0..<capacity {
            out.append(contentsOf: storage[(head + i) % capacity])
        }
        return out
    }
}
