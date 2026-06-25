package ai.zetic.aiberry.face

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import ai.zetic.aiberry.core.AppConfig
import java.util.concurrent.Executors

/**
 * Front-camera capture via CameraX: a mirrored [PreviewView] for the picture-in-picture, plus an
 * [ImageAnalysis] stream throttled to [AppConfig.Face.INFERENCE_HZ] and forwarded to the face
 * model. Mirrors iOS-Aiberry's `CameraController` (front camera, ~3 Hz, latest-frame only).
 */
class CameraController(private val context: Context) {
    private var provider: ProcessCameraProvider? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "aiberry-camera").apply { isDaemon = true }
    }
    @Volatile private var lastForwardMs = 0L

    /** Bind preview + analysis to [owner]; call on the main thread. */
    fun bind(owner: LifecycleOwner, previewView: PreviewView, onFrame: (Bitmap, Int) -> Unit) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val cameraProvider = future.get()
            provider = cameraProvider

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                try {
                    val now = System.currentTimeMillis()
                    if (now - lastForwardMs >= AppConfig.Face.frameIntervalMs) {
                        lastForwardMs = now
                        val bitmap = imageProxy.toBitmap()
                        onFrame(bitmap, imageProxy.imageInfo.rotationDegrees)
                    }
                } catch (_: Throwable) {
                    // Skip this frame.
                } finally {
                    imageProxy.close()
                }
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    owner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    analysis,
                )
            } catch (_: Throwable) {
                // Camera unavailable (e.g. permission just revoked); leave unbound.
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun unbind() {
        try {
            provider?.unbindAll()
        } catch (_: Throwable) {
        }
        provider = null
    }
}
