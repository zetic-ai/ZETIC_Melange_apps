package ai.zetic.demo.posemotion.video

import android.os.Handler
import android.os.SystemClock

/**
 * Feeds decoded frames to the inference thread. Two pacing modes mirroring iOS:
 * - BENCHMARK: the next frame is decoded only after the previous one finishes
 *   processing (self-reposting step), so measured FPS = sustainable throughput.
 * - REALTIME: frames are paced to their presentation timestamps against the wall
 *   clock; frames more than 50 ms late are dropped.
 * Loops at end of stream. All work happens on `handler`'s thread.
 */
class ClipFrameSource(
    private val decoder: VideoFrameDecoder,
    private val mode: Mode,
    private val handler: Handler,
) {
    enum class Mode(val label: String) { BENCHMARK("Benchmark"), REALTIME("Realtime") }

    /** Called on the handler thread for every frame to process. */
    var onFrame: ((DecodedFrame) -> Unit)? = null

    private var running = false
    private var wallStartNs = 0L

    fun start() {
        handler.post {
            if (running) return@post
            running = true
            decoder.open()
            wallStartNs = System.nanoTime()
            step()
        }
    }

    fun stop() {
        handler.post {
            running = false
            decoder.release()
        }
    }

    private fun step() {
        handler.post {
            if (!running) return@post

            val frame = decoder.nextFrame()
            if (frame == null) {
                decoder.rewind()          // end of clip: loop
                wallStartNs = System.nanoTime()
                if (running) step()
                return@post
            }

            if (mode == Mode.REALTIME) {
                val dueNs = wallStartNs + frame.ptsUs * 1000
                val nowNs = System.nanoTime()
                if (nowNs < dueNs) {
                    SystemClock.sleep((dueNs - nowNs) / 1_000_000)
                } else if (nowNs - dueNs > 50_000_000) {
                    step()                // more than a frame late: drop to catch up
                    return@post
                }
            }

            onFrame?.invoke(frame)
            step()
        }
    }
}
