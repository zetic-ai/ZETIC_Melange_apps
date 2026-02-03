package com.zeticai.yolov26.ui

import android.graphics.Bitmap
import android.net.Uri
import android.view.TextureView
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import com.zeticai.yolov26.YOLOv26Model
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

@OptIn(UnstableApi::class) 
@Composable
fun VideoScreen(model: YOLOv26Model) {
    var videoUri by remember { mutableStateOf<Uri?>(null) }
    val context = LocalContext.current
    
    // State for detection
    val boxes by model.detectionResults.collectAsState()
    val sourceSize by model.sourceImageSize.collectAsState()
    
    // Video Picker
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri != null) {
            videoUri = uri
            // Clear previous boxes ?
        }
    }
    
    Box(modifier = Modifier.fillMaxSize()) {
        if (videoUri != null) {
            VideoPlayerAndDetector(uri = videoUri!!, model = model)
            
            // Overlay is now inside VideoPlayerAndDetector for better alignment
            
        } else {
            Text("Select a video", modifier = Modifier.align(Alignment.Center))
        }
        
        Button(
            onClick = { 
                launcher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
            },
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 60.dp)
        ) {
            Text("Pick Video")
        }
    }
}

@OptIn(UnstableApi::class)
@Composable
fun VideoPlayerAndDetector(uri: Uri, model: YOLOv26Model) {
    val context = LocalContext.current
    
    // ExoPlayer Instance
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            repeatMode = Player.REPEAT_MODE_ONE
            playWhenReady = true
        }
    }
    
    // State for Aspect Ratio
    var videoAspectRatio by remember { mutableStateOf(16f / 9f) } // Default
    var textureView: TextureView? by remember { mutableStateOf(null) }
    
    // Setup Player
    LaunchedEffect(uri) {
        exoPlayer.setMediaItem(MediaItem.fromUri(uri))
        exoPlayer.prepare()
        
        // Listener for Aspect Ratio
        val listener = object : Player.Listener {
            override fun onVideoSizeChanged(videoSize: androidx.media3.common.VideoSize) {
                if (videoSize.width > 0 && videoSize.height > 0) {
                     val ratio = videoSize.width.toFloat() / videoSize.height.toFloat()
                     // Handle Rotation
                     if (videoSize.unappliedRotationDegrees == 90 || videoSize.unappliedRotationDegrees == 270) {
                         videoAspectRatio = 1f / ratio
                     } else {
                         videoAspectRatio = ratio
                     }
                }
            }
        }
        exoPlayer.addListener(listener)
    }
    
    // Cleanup
    DisposableEffect(Unit) {
        onDispose {
            exoPlayer.release()
        }
    }
    
    // Detector Loop
    LaunchedEffect(textureView) {
        if (textureView != null) {
            while (isActive) {
                delay(33) // ~30 FPS
                
                val view = textureView
                if (view != null && exoPlayer.isPlaying) {
                    val bitmap = view.bitmap 
                    if (bitmap != null) {
                         model.detect(bitmap)
                    }
                }
            }
        }
    }
    
    // Layout: Centered Box with aspect ratio of the video
    // This allows BoundingBoxOverlay to map perfectly to the video bounds
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .aspectRatio(videoAspectRatio)
                .fillMaxSize() 
        ) {
            // 1. Video Layer
            AndroidView(
                factory = { ctx ->
                    val tex = TextureView(ctx).apply {
                        layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                        )
                    }
                    exoPlayer.setVideoTextureView(tex)
                    textureView = tex
                    tex
                },
                modifier = Modifier.fillMaxSize()
            )
            
            // 2. Overlay Layer (Inside the Video Box)
            // We can now use 'isFill = true' (Fill Parent) since the Parent IS the Video Frame
            val boxes by model.detectionResults.collectAsState()
            val sourceSize by model.sourceImageSize.collectAsState()
            
            // Note: pass isFill=true to mimic "Match Parent" logic in overlay
            // or pass isFill=false (Fit) - since parent Aspect Ratio matches Source Ratio, Fit == Fill.
            // Using logic from Components.kt: SourceWidth/Height logic still applies.
            BoundingBoxOverlay(
                boxes = boxes,
                sourceWidth = sourceSize.first,
                sourceHeight = sourceSize.second,
                isFill = true 
            )
        }
    }
}


// Remove Helper as it is no longer needed (we have direct ref)
