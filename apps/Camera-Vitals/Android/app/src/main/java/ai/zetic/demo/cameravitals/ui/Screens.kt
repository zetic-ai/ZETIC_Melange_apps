package ai.zetic.demo.cameravitals.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import ai.zetic.demo.cameravitals.state.MeasurementState
import ai.zetic.demo.cameravitals.state.VitalsViewModel

/** Hosts the CameraX PreviewView and binds the camera once. */
@Composable
fun CameraPreview(vm: VitalsViewModel, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    LaunchedEffect(Unit) {
        vm.camera.start(lifecycleOwner, previewView.surfaceProvider)
    }
    AndroidView(factory = { previewView }, modifier = modifier)
}

/** Routes between screens by FSM state and overlays the report sheet. */
@Composable
fun RootScreen(vm: VitalsViewModel = viewModel()) {
    val context = LocalContext.current
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> vm.onPermissionResult(granted) }

    LaunchedEffect(Unit) {
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) vm.onPermissionResult(true) else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    Box(
        Modifier
            .fillMaxSize()
            .background(Theme.background)
            .windowInsetsPadding(WindowInsets.systemBars)   // keep content out of the status/nav bars
    ) {
        when (val s = vm.state) {
            is MeasurementState.LoadingModel -> DownloadScreen(s.progress)
            is MeasurementState.PermissionDenied -> PermissionScreen()
            is MeasurementState.ErrorState -> ErrorScreen(s.message) { vm.retry() }
            is MeasurementState.Warmup, is MeasurementState.Live -> MeasureScreen(vm)
        }
    }

    vm.report?.let { report ->
        ReportSheet(report = report, onDismiss = { vm.dismissReport() })
    }
}

@Composable
fun DownloadScreen(progress: Float) {
    Column(
        Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(contentAlignment = Alignment.Center) {
            CircularProgressIndicator(
                progress = { if (progress > 0f) progress else 0.02f },
                modifier = Modifier.size(120.dp),
                color = Theme.accent,
                strokeWidth = 10.dp,
                trackColor = Theme.accentSoft
            )
            Icon(Icons.Filled.Memory, null, tint = Theme.accent, modifier = Modifier.size(36.dp))
        }
        androidx.compose.foundation.layout.Spacer(Modifier.size(26.dp))
        Text("Preparing on-device model", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Theme.textPrimary)
        Text(
            if (progress > 0f) "${(progress * 100).toInt()}%" else "Optimizing for the NPU…",
            fontSize = 15.sp, color = Theme.textSecondary
        )
    }
}

@Composable
fun PermissionScreen() {
    val context = LocalContext.current
    CenteredMessage(
        icon = Icons.Filled.Warning,
        title = "Camera access needed",
        body = "Camera Vitals reads your pulse from the front camera. Video never leaves your phone.",
        buttonText = "Open Settings"
    ) {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
        context.startActivity(intent)
    }
}

@Composable
fun ErrorScreen(message: String, onRetry: () -> Unit) {
    CenteredMessage(
        icon = Icons.Filled.Warning,
        title = "Something went wrong",
        body = message,
        buttonText = "Try again",
        onClick = onRetry
    )
}

@Composable
private fun CenteredMessage(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    body: String,
    buttonText: String,
    onClick: () -> Unit
) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(icon, null, tint = Theme.accent, modifier = Modifier.size(44.dp))
        androidx.compose.foundation.layout.Spacer(Modifier.size(16.dp))
        Text(title, fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Theme.textPrimary)
        androidx.compose.foundation.layout.Spacer(Modifier.size(8.dp))
        Text(body, fontSize = 15.sp, color = Theme.textSecondary, textAlign = TextAlign.Center)
        androidx.compose.foundation.layout.Spacer(Modifier.size(20.dp))
        Button(
            onClick = onClick,
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(containerColor = Theme.accent, contentColor = androidx.compose.ui.graphics.Color.White),
            contentPadding = PaddingValues(horizontal = 28.dp, vertical = 12.dp)
        ) {
            Text(buttonText, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}
