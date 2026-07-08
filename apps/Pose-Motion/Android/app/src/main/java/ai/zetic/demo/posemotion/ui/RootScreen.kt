package ai.zetic.demo.posemotion.ui

import ai.zetic.demo.posemotion.state.DemoViewModel
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue

@Composable
fun RootScreen(viewModel: DemoViewModel) {
    val phase by viewModel.phase.collectAsState()
    when (phase) {
        DemoViewModel.Phase.LOADING -> ModelDownloadScreen(viewModel)
        DemoViewModel.Phase.MISSING_CLIP -> MissingClipScreen(viewModel)
        DemoViewModel.Phase.RUNNING -> DemoScreen(viewModel)
    }
}
