package ai.zetic.demo.imageto3d

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

/** Orchestrates pick → preprocess → depth inference → colormap + relief mesh. */
class AppViewModel(application: Application) : AndroidViewModel(application) {
    sealed interface Phase {
        data class LoadingModel(val progress: Float) : Phase
        data object Idle : Phase
        data class Processing(val stage: String) : Phase
        data object Ready : Phase
        data class Error(val message: String) : Phase
    }

    class Latency {
        var modelLoadMs = 0.0
        var depthMs = 0.0
        var meshMs = 0.0
    }

    var phase by mutableStateOf<Phase>(Phase.LoadingModel(0f))
        private set
    var latency by mutableStateOf(Latency())
        private set
    var photo by mutableStateOf<Bitmap?>(null)
        private set
    var depthImage by mutableStateOf<Bitmap?>(null)
        private set
    var mesh by mutableStateOf<MeshData?>(null)
        private set
    var texture by mutableStateOf<Bitmap?>(null)
        private set
    var showPoints by mutableStateOf(false)

    private val model = DepthModel()
    private var pending: Bitmap? = null

    fun loadModel() {
        if (phase !is Phase.LoadingModel) return
        viewModelScope.launch(Dispatchers.IO) {
            val t0 = System.nanoTime()
            try {
                model.load(getApplication()) { p ->
                    viewModelScope.launch { phase = Phase.LoadingModel(p) }
                }
                withContext(Dispatchers.Main) {
                    latency.modelLoadMs = (System.nanoTime() - t0) / 1e6
                    phase = Phase.Idle
                    pending?.let { process(it) }
                    pending = null
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    phase = Phase.Error("Model load failed: ${e.message}")
                }
            }
        }
    }

    fun retryLoad() {
        phase = Phase.LoadingModel(0f)
        loadModel()
    }

    fun processUri(uri: Uri) {
        val context = getApplication<Application>()
        val bitmap = ImageDecoder.decodeBitmap(
            ImageDecoder.createSource(context.contentResolver, uri)
        ) { decoder, _, _ -> decoder.isMutableRequired = true }
        process(bitmap)
    }

    fun process(image: Bitmap) {
        if (phase is Phase.LoadingModel) {
            pending = image
            return
        }
        phase = Phase.Processing("Preprocessing…")
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val input = ImagePreprocessor.prepare(image)
                withContext(Dispatchers.Main) { phase = Phase.Processing("Running depth model…") }

                val depth = model.infer(input.chw)
                withContext(Dispatchers.Main) { phase = Phase.Processing("Building 3D mesh…") }

                val t1 = System.nanoTime()
                val meshData = DepthTo3D.build(depth, input.texture)
                val meshMs = (System.nanoTime() - t1) / 1e6
                val depthBitmap = DepthColormap.bitmap(depth)

                withContext(Dispatchers.Main) {
                    photo = input.texture
                    depthImage = depthBitmap
                    mesh = meshData
                    texture = input.texture
                    latency.depthMs = model.lastInferMs
                    latency.meshMs = meshMs
                    phase = Phase.Ready
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    phase = Phase.Error("Inference failed: ${e.message}")
                }
            }
        }
    }

    /** Headless verification: process the bundled sample and write results to
     *  external files for `adb pull`. Triggered by `--ez selftest true`. */
    fun runSelfTest() {
        viewModelScope.launch(Dispatchers.IO) {
            val context = getApplication<Application>()
            val dir = File(context.getExternalFilesDir(null), "selftest").apply { mkdirs() }
            val pushed = File(context.getExternalFilesDir(null), "selftest_input.jpg")
            val bitmap = if (pushed.exists()) {
                BitmapFactory.decodeFile(pushed.path)
            } else {
                context.assets.open("sample.jpg").use { BitmapFactory.decodeStream(it) }
            }
            if (bitmap == null) {
                File(dir, "stats.json").writeText("""{"status":"no test image"}""")
                return@launch
            }
            withContext(Dispatchers.Main) { process(bitmap) }

            // Poll until processing finishes, then dump results.
            while (phase !is Phase.Ready && phase !is Phase.Error) {
                kotlinx.coroutines.delay(250)
            }
            val stats = JSONObject()
            when (val p = phase) {
                is Phase.Ready -> {
                    stats.put("status", "ok")
                    stats.put("depthInferMs", latency.depthMs.toInt())
                    stats.put("meshMs", latency.meshMs.toInt())
                    stats.put("triangles", (mesh?.triangleIndices?.size ?: 0) / 3)
                    stats.put("points", mesh?.pointIndices?.size ?: 0)
                    photo?.let { save(it, File(dir, "photo.png")) }
                    depthImage?.let { save(it, File(dir, "depth.png")) }
                }
                is Phase.Error -> stats.put("status", p.message)
                else -> stats.put("status", "unknown")
            }
            File(dir, "stats.json").writeText(stats.toString())
            android.util.Log.i("ImageTo3D", "selftest finished: $stats")
        }
    }

    private fun save(bitmap: Bitmap, file: File) {
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    override fun onCleared() {
        model.deinit()
    }
}
