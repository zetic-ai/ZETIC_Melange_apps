package ai.zetic.demo.posemotion.video

import ai.zetic.demo.posemotion.AppConfig
import android.content.Context
import android.content.res.AssetFileDescriptor
import java.io.File

sealed class ClipSource {
    data class Asset(val afd: AssetFileDescriptor) : ClipSource()
    data class Local(val path: String) : ClipSource()
}

/** Mirrors the iOS lookup order per clip: bundled asset first, then the app files dir
 *  (`adb push <clip> /sdcard/Android/data/ai.zetic.demo.posemotion/files/`). */
object ClipLocator {
    fun locate(context: Context, fileName: String): ClipSource? {
        runCatching {
            return ClipSource.Asset(context.assets.openFd(fileName))
        }
        val file = File(context.getExternalFilesDir(null), fileName)
        if (file.exists()) return ClipSource.Local(file.absolutePath)
        return null
    }

    /** The configured clips that actually resolve on this device, in config order. */
    fun availableClips(context: Context): List<String> =
        AppConfig.CLIP_FILES.filter { name ->
            val found = locate(context, name)
            ((found as? ClipSource.Asset)?.afd)?.close()
            found != null
        }
}
