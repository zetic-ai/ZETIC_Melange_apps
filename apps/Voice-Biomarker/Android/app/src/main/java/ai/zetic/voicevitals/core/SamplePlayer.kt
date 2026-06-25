package ai.zetic.voicevitals.core

import android.content.Context
import android.media.MediaPlayer
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * Plays a bundled sample clip aloud through the speaker. Intended use: start a
 * recording, play a sample so the microphone picks it up, then stop the recording
 * to analyze it — an acoustic loopback through the live mic pipeline.
 */
class SamplePlayer {
    /** Key of the clip currently playing, or null when idle. */
    var playing by mutableStateOf<String?>(null)
        private set

    private var player: MediaPlayer? = null

    /**
     * Play a bundled raw resource aloud. [key] identifies it for the [playing] state.
     * No-op if the resource can't be loaded.
     */
    fun play(context: Context, resId: Int, key: String) {
        val p = MediaPlayer.create(context, resId) ?: return
        player?.release()
        player = p
        playing = key
        p.setOnCompletionListener { stop() }
        p.start()
    }

    fun stop() {
        player?.let {
            try {
                it.stop()
            } catch (_: Throwable) {
            }
            it.release()
        }
        player = null
        playing = null
    }
}
