package com.zeticai.yolov26

import android.Manifest
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.zeticai.yolov26.ui.BoundingBoxOverlay
import com.zeticai.yolov26.ui.CameraScreen
import com.zeticai.yolov26.ui.VideoScreen
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        val model = YOLOv26Model(this)
        
        setContent {
            MaterialTheme {
                MainScreen(model)
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun MainScreen(model: YOLOv26Model) {
    var selectedTab by remember { mutableStateOf(0) }
    val tabs = listOf("Photo", "Video", "Camera")
    
    val isModelLoaded by model.isModelLoaded.collectAsState()
    
    if (!isModelLoaded) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Loading Model...")
            }
        }
        return
    }

    Scaffold(
        topBar = {
            Column {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.primaryContainer)
                        .padding(vertical = 4.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "(Powered by MLange)",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }
                TabRow(selectedTabIndex = selectedTab) {
                    tabs.forEachIndexed { index, title ->
                        Tab(
                            text = { Text(title) },
                            selected = selectedTab == index,
                            onClick = { selectedTab = index }
                        )
                    }
                }
            }
        }
    ) { paddingValues ->
        Box(modifier = Modifier.padding(paddingValues).fillMaxSize()) {
            when (selectedTab) {
                0 -> PhotoScreen(model)
                1 -> VideoScreen(model)
                2 -> {
                    // Camera Permission
                    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)
                    if (cameraPermissionState.status.isGranted) {
                        CameraScreen(model)
                    } else {
                        Column(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.Center,
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text("Camera permission required")
                            Button(onClick = { cameraPermissionState.launchPermissionRequest() }) {
                                Text("Grant Permission")
                            }
                        }
                    }
                }
            }
            
            // Debug Info
            val inferenceTime by model.inferenceTime.collectAsState()
            Text(
                text = "Total process time: ${inferenceTime}ms",
                color = Color.Red,
                modifier = Modifier.align(Alignment.BottomStart).padding(16.dp)
            )
        }
    }
}

@Composable
fun PhotoScreen(model: YOLOv26Model) {
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    val boxes by model.detectionResults.collectAsState()
    
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        imageUri = uri
        uri?.let {
            scope.launch {
                val inputStream = context.contentResolver.openInputStream(it)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                model.detect(bitmap)
            }
        }
    }
    
    Box(modifier = Modifier.fillMaxSize()) {
        if (imageUri != null) {
            // Display Image
            // Note: Efficient loading omitted for brevity
             val inputStream = context.contentResolver.openInputStream(imageUri!!)
             val bitmap = BitmapFactory.decodeStream(inputStream)
             
             Image(
                 bitmap = bitmap.asImageBitmap(),
                 contentDescription = "Selected Image",
                 modifier = Modifier.fillMaxSize()
             )
             
             // Scale factors?
             // Assuming Image fills screen or we match aspect ratio.
             // For strict correctness we need to know rendered size vs bitmap size.
             // We'll trust AspectFit logic of Compose vs BoundingBoxOverlay
             
             val sourceSize by model.sourceImageSize.collectAsState()
             
             BoundingBoxOverlay(
                boxes = boxes,
                sourceWidth = sourceSize.first,
                sourceHeight = sourceSize.second,
                isFill = false
             )
             
        } else {
            Text("Select an image", modifier = Modifier.align(Alignment.Center))
        }
        
        Button(
            onClick = { launcher.launch("image/*") },
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 60.dp)
        ) {
            Text("Pick Image")
        }
    }
}
