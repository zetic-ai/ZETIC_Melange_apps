package ai.zetic.demo.imageto3d

import android.Manifest
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import java.io.File

class MainActivity : ComponentActivity() {
    private val viewModel: AppViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel.loadModel()
        if (intent.getBooleanExtra("selftest", false)) {
            viewModel.runSelfTest()
        }
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                RootScreen(viewModel)
            }
        }
    }
}

@Composable
private fun RootScreen(vm: AppViewModel) {
    val context = LocalContext.current
    var cameraUri by remember { mutableStateOf<Uri?>(null) }

    val pickMedia = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let { vm.processUri(it) } }

    val takePicture = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success -> if (success) cameraUri?.let { vm.processUri(it) } }

    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            val uri = createCaptureUri(context)
            cameraUri = uri
            takePicture.launch(uri)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF080808))
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Image to 3D", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Color.White)
        Text(
            "On-device depth analysis · ZETIC.MLange",
            fontSize = 12.sp, color = Color.Gray,
        )
        Spacer(Modifier.height(12.dp))

        when (val phase = vm.phase) {
            is AppViewModel.Phase.LoadingModel -> CenterStatus {
                LinearProgressIndicator(
                    progress = { phase.progress },
                    modifier = Modifier.width(220.dp),
                )
                Spacer(Modifier.height(10.dp))
                Text(
                    if (phase.progress > 0f)
                        "Downloading model… ${(phase.progress * 100).toInt()}%"
                    else "Preparing model…",
                    fontSize = 13.sp, color = Color.Gray,
                )
            }
            is AppViewModel.Phase.Idle -> CenterStatus {
                Text(
                    "Pick or take a photo to analyze its depth in 3D — fully offline.",
                    fontSize = 13.sp, color = Color.Gray, textAlign = TextAlign.Center,
                    modifier = Modifier.width(260.dp),
                )
            }
            is AppViewModel.Phase.Processing -> CenterStatus {
                CircularProgressIndicator()
                Spacer(Modifier.height(10.dp))
                Text(phase.stage, fontSize = 13.sp, color = Color.Gray)
            }
            is AppViewModel.Phase.Ready -> ResultView(vm, Modifier.weight(1f))
            is AppViewModel.Phase.Error -> CenterStatus {
                Text(
                    phase.message, fontSize = 13.sp, color = Color(0xFFFFA726),
                    textAlign = TextAlign.Center, modifier = Modifier.width(280.dp),
                )
                Spacer(Modifier.height(10.dp))
                OutlinedButton(onClick = { vm.retryLoad() }) { Text("Retry model load") }
            }
        }

        if (vm.phase !is AppViewModel.Phase.Ready) Spacer(Modifier.weight(1f))
        Spacer(Modifier.height(12.dp))

        val interactive = vm.phase is AppViewModel.Phase.Idle ||
            vm.phase is AppViewModel.Phase.Ready ||
            vm.phase is AppViewModel.Phase.Error
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                onClick = {
                    pickMedia.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
                enabled = interactive,
                modifier = Modifier.weight(1f),
            ) { Text("Photo Library") }
            OutlinedButton(
                onClick = { cameraPermission.launch(Manifest.permission.CAMERA) },
                enabled = interactive,
                modifier = Modifier.weight(1f),
            ) { Text("Camera") }
        }
    }
}

@Composable
private fun ColumnScope.CenterStatus(content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier.weight(1f).fillMaxWidth(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
        content = content,
    )
}

@Composable
private fun ResultView(vm: AppViewModel, modifier: Modifier) {
    Column(modifier = modifier.fillMaxWidth()) {
        // Top: compact interactive 3D relief with mode toggle overlaid.
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(230.dp)
                .clip(RoundedCornerShape(12.dp)),
            contentAlignment = Alignment.BottomCenter,
        ) {
            val mesh = vm.mesh
            val texture = vm.texture
            if (mesh != null && texture != null) {
                ReliefPane(mesh, texture, vm.showPoints)
            }
            SingleChoiceSegmentedButtonRow(Modifier.padding(bottom = 8.dp).width(180.dp)) {
                SegmentedButton(
                    selected = !vm.showPoints,
                    onClick = { vm.showPoints = false },
                    shape = RoundedCornerShape(topStart = 16.dp, bottomStart = 16.dp),
                ) { Text("Mesh", fontSize = 12.sp) }
                SegmentedButton(
                    selected = vm.showPoints,
                    onClick = { vm.showPoints = true },
                    shape = RoundedCornerShape(topEnd = 16.dp, bottomEnd = 16.dp),
                ) { Text("Points", fontSize = 12.sp) }
            }
        }

        Spacer(Modifier.height(12.dp))

        // Main display: photo → depth analysis, side by side, with latency HUD.
        Box(Modifier.weight(1f).fillMaxWidth()) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Pane(vm.photo, "Photo", Modifier.weight(1f))
                Pane(vm.depthImage, "Depth", Modifier.weight(1f))
            }
            LatencyHUD(vm.latency, Modifier.align(Alignment.TopEnd).padding(6.dp))
        }
    }
}

@Composable
private fun ReliefPane(mesh: MeshData, texture: Bitmap, showPoints: Boolean) {
    AndroidView(
        factory = { context -> Relief3DView(context) },
        update = { view ->
            view.setMesh(mesh, texture)
            view.setMode(showPoints)
        },
        modifier = Modifier.fillMaxSize(),
    )
}

@Composable
private fun Pane(bitmap: Bitmap?, label: String, modifier: Modifier) {
    Column(modifier = modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        if (bitmap != null) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = label,
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp)),
            )
        } else {
            Box(
                Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0xFF1F1F1F))
            )
        }
        Spacer(Modifier.height(6.dp))
        Text(label, fontSize = 12.sp, color = Color.Gray)
    }
}

@Composable
private fun LatencyHUD(latency: AppViewModel.Latency, modifier: Modifier) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0xCC202020))
            .padding(8.dp),
    ) {
        HudRow("load", latency.modelLoadMs)
        HudRow("depth", latency.depthMs)
        HudRow("mesh", latency.meshMs)
    }
}

@Composable
private fun HudRow(label: String, ms: Double) {
    Row(Modifier.width(110.dp), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, fontSize = 11.sp, color = Color.Gray, fontFamily = FontFamily.Monospace)
        Text(
            if (ms >= 1000) "%.1f s".format(ms / 1000) else "%.0f ms".format(ms),
            fontSize = 11.sp, color = Color.White, fontFamily = FontFamily.Monospace,
        )
    }
}

private fun createCaptureUri(context: android.content.Context): Uri {
    val dir = File(context.cacheDir, "captures").apply { mkdirs() }
    val file = File.createTempFile("capture_", ".jpg", dir)
    return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
}
