package ai.zetic.aiberry.ui

import android.graphics.Bitmap
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import ai.zetic.aiberry.face.CameraController

/**
 * Mirrored front-camera preview. Binds [controller] to the current lifecycle ONCE while shown
 * and forwards throttled analysis frames to [onFrame] (the face model); unbinds on dispose.
 * Binding is keyed on the view + lifecycle so the mic-meter recompositions don't rebind.
 */
@Composable
fun CameraPreview(
    controller: CameraController,
    onFrame: (Bitmap, Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    AndroidView(factory = { previewView }, modifier = modifier)
    LaunchedEffect(previewView, lifecycleOwner) {
        controller.bind(lifecycleOwner, previewView, onFrame)
    }
    DisposableEffect(Unit) {
        onDispose { controller.unbind() }
    }
}
