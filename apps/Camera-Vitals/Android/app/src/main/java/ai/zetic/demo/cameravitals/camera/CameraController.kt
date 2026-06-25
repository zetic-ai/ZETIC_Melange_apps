package ai.zetic.demo.cameravitals.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors

/**
 * Owns the CameraX session: front camera, RGBA frames delivered to `onFrame` as an upright
 * Bitmap on a dedicated analyzer thread. Keeps the per-frame work tiny so capture never stalls.
 * The pipeline that receives `onFrame` is responsible for recycling the bitmap.
 */
class CameraController(private val context: Context) {
    var onFrame: ((Bitmap) -> Unit)? = null

    private var provider: ProcessCameraProvider? = null
    private val analyzerExecutor = Executors.newSingleThreadExecutor()

    fun start(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val cameraProvider = future.get()
            provider = cameraProvider

            val preview = Preview.Builder().build().also { it.setSurfaceProvider(surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(analyzerExecutor, ::handleFrame) }

            val selector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
            } catch (_: Exception) {
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun handleFrame(proxy: ImageProxy) {
        try {
            val rotation = proxy.imageInfo.rotationDegrees
            val raw = proxy.toBitmap()
            val upright = if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
                    .also { if (it !== raw) raw.recycle() }
            } else {
                raw
            }
            onFrame?.invoke(upright)
        } catch (_: Exception) {
        } finally {
            proxy.close()
        }
    }

    fun stop() {
        provider?.unbindAll()
    }

    fun shutdown() {
        provider?.unbindAll()
        analyzerExecutor.shutdown()
    }
}
