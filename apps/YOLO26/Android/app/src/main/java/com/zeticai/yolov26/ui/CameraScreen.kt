package com.zeticai.yolov26.ui

import android.content.Context
import android.util.Log
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.core.AspectRatio // Added for correct ratio
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.zeticai.yolov26.YOLOv26Model
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

@Composable
fun CameraScreen(model: YOLOv26Model) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    val boxes by model.detectionResults.collectAsState()
    
    // Remember PreviewView so it doesn't recreate on recomposition
    val previewView = remember { PreviewView(context) }
    
    // Setup Camera only once when lifecycleOwner changes
    LaunchedEffect(lifecycleOwner) {
        setupCamera(context, lifecycleOwner, previewView, model)
    }
    
    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { 
                previewView.apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FIT_CENTER
                }
            },
            modifier = Modifier.fillMaxSize()
        )
        
        val sourceSize by model.sourceImageSize.collectAsState()
        
        BoundingBoxOverlay(
            boxes = boxes,
            sourceWidth = sourceSize.first,
            sourceHeight = sourceSize.second,
            isFill = false, // Match FIT_CENTER (Letterbox)
            alignmentTopStart = false
        )
    }
}

private fun setupCamera(
    context: Context,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    previewView: PreviewView,
    model: YOLOv26Model
) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    val isBusy = AtomicBoolean(false)
    
    cameraProviderFuture.addListener({
        val cameraProvider = cameraProviderFuture.get()
        
        // Correct Rotation
        val rotation = previewView.display.rotation
        
        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3) 
            .setTargetRotation(rotation)
             .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            
        imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
            processImageProxy(imageProxy, model, isBusy)
        }
        
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        
        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )
        } catch (exc: Exception) {
            Log.e("CameraScreen", "Use case binding failed", exc)
        }
        
    }, ContextCompat.getMainExecutor(context))
}

@androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class) 
private fun processImageProxy(
    imageProxy: ImageProxy, 
    model: YOLOv26Model,
    isBusy: AtomicBoolean
) {
    if (isBusy.compareAndSet(false, true)) {
        // Fast Path: Convert to Bitmap
        val rawBitmap = imageProxy.toBitmap()
        var bitmap = rawBitmap
        
        // Manual Rotation Fix if needed (Some devices/versions of CameraX behave differently with targetRotation)
        val rotation = imageProxy.imageInfo.rotationDegrees
        if (rotation != 0) {
            val matrix = android.graphics.Matrix()
            matrix.postRotate(rotation.toFloat())
            bitmap = android.graphics.Bitmap.createBitmap(
                rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true
            )
        }
        
        imageProxy.close()
        
        CoroutineScope(Dispatchers.Default).launch {
            try {
                model.detect(bitmap)
            } finally {
                isBusy.set(false)
            }
        }
    } else {
        // Drop Frame
        imageProxy.close()
    }
}

