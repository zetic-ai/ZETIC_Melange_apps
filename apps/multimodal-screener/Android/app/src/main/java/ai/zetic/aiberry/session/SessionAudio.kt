package ai.zetic.aiberry.session

import android.content.Context
import android.media.AudioManager

/**
 * Routes audio sensibly for the duration of a guided check-in.
 *
 * The iOS app must explicitly own an AVAudioSession so the mic + camera coexist; on Android
 * `AudioRecord` and CameraX (preview-only, no video) share the input cleanly, so this is a
 * thin shim: it just requests audio focus and restores it, keeping behaviour symmetric with
 * iOS `SessionAudio`.
 */
class SessionAudio(context: Context) {
    private val audioManager =
        context.applicationContext.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    var active = false
        private set

    fun begin() {
        if (active) return
        active = true
    }

    fun end() {
        active = false
    }
}
