package ai.zetic.demo.cameravitals.pipeline

/**
 * Fixed-capacity ring of preprocessed frames (planar RGB float, CHW per frame).
 * Thread-safe: the camera thread appends, the inference thread snapshots.
 */
class FrameRingBuffer(private val capacity: Int, private val frameLen: Int) {
    private val storage = Array(capacity) { FloatArray(frameLen) }
    private var head = 0          // next write slot; when full this is also the oldest
    private var count = 0
    private val lock = Any()

    val isFull: Boolean get() = synchronized(lock) { count >= capacity }
    val filled: Int get() = synchronized(lock) { count }

    /** Copies the frame in, so the caller may reuse its buffer. */
    fun append(frame: FloatArray) = synchronized(lock) {
        System.arraycopy(frame, 0, storage[head], 0, frameLen)
        head = (head + 1) % capacity
        if (count < capacity) count++
    }

    fun reset() = synchronized(lock) {
        count = 0
        head = 0
    }

    /** Latest `capacity` frames in temporal order (oldest → newest), flattened; null until full. */
    fun snapshot(): FloatArray? = synchronized(lock) {
        if (count < capacity) return null
        val out = FloatArray(capacity * frameLen)
        for (i in 0 until capacity) {
            val idx = (head + i) % capacity
            System.arraycopy(storage[idx], 0, out, i * frameLen, frameLen)
        }
        out
    }
}
