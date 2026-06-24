package ai.zetic.aiberry

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import ai.zetic.aiberry.session.CheckInViewModel
import ai.zetic.aiberry.session.Phase
import ai.zetic.aiberry.ui.AnalyzingScreen
import ai.zetic.aiberry.ui.ConversationScreen
import ai.zetic.aiberry.ui.InsightsScreen
import ai.zetic.aiberry.ui.IntroScreen
import ai.zetic.aiberry.ui.LandingScreen
import ai.zetic.aiberry.ui.Theme

class MainActivity : ComponentActivity() {

    private val vm: CheckInViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val permissionLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.RequestMultiplePermissions(),
            ) { /* models still preload regardless; UI gates on permission via system. */ }

            LaunchedEffect(Unit) {
                permissionLauncher.launch(
                    arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO),
                )
            }
            LaunchedEffect(Unit) { vm.preloadAll() }

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Theme.Bg)
                    .windowInsetsPadding(WindowInsets.safeDrawing),
            ) {
                when (val phase = vm.phase) {
                    is Phase.Idle -> LandingScreen(vm, onStart = { vm.showIntro() })
                    is Phase.Intro -> IntroScreen(vm)
                    is Phase.Question -> ConversationScreen(vm)
                    is Phase.Analyzing -> AnalyzingScreen()
                    is Phase.Insights -> InsightsScreen(vm, phase.report)
                }
            }
        }
    }
}
