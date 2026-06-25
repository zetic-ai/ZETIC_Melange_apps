package ai.zetic.voicevitals.core

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import ai.zetic.voicevitals.emotion.EmotionModel
import ai.zetic.voicevitals.respiratory.YamnetModel

/**
 * Owns the on-device models for the app's lifetime and preloads them at launch so
 * the NPU models are downloaded/compiled before the user records.
 */
class AppViewModel(app: Application) : AndroidViewModel(app) {
    val emotion = EmotionModel(app)
    val yamnet = YamnetModel(app)

    private var preloaded = false

    /** Download + compile every model up front (each runs on its own background thread). */
    fun preloadAll() {
        if (preloaded) return
        preloaded = true
        emotion.preload()
        yamnet.preload()
    }
}
