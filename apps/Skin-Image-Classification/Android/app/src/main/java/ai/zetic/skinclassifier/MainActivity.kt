package ai.zetic.skinclassifier

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import ai.zetic.skinclassifier.state.AnalysisState
import ai.zetic.skinclassifier.state.DiagnosisViewModel
import ai.zetic.skinclassifier.ui.CaptureScreen
import ai.zetic.skinclassifier.ui.DownloadScreen
import ai.zetic.skinclassifier.ui.ResultsScreen
import ai.zetic.skinclassifier.ui.Theme

class MainActivity : ComponentActivity() {

    private val vm: DiagnosisViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            LaunchedEffect(Unit) { vm.bootstrap() }

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Theme.Bg)
                    .windowInsetsPadding(WindowInsets.safeDrawing),
            ) {
                when {
                    !vm.canAnalyze -> DownloadScreen(vm)
                    vm.analysis is AnalysisState.None -> CaptureScreen(vm)
                    else -> ResultsScreen(vm)
                }
            }
        }
    }
}
