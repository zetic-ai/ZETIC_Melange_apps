package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.AppModel
import ai.zetic.demo.cherrypad.data.Prefs
import ai.zetic.demo.cherrypad.llm.LLMService
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch

@Composable
fun RootScreen(model: AppModel = viewModel()) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val phase by LLMService.phase.collectAsState()

    var showSettings by remember { mutableStateOf(false) }
    var showOnboarding by remember { mutableStateOf(!Prefs.hasSeenOnboarding(context)) }

    // Warm / download the model at launch.
    LaunchedEffect(Unit) {
        runCatching { LLMService.ensureLoaded(context) }
    }

    ComposeScreen(
        model = model,
        phase = phase,
        onOpenSettings = { showSettings = true },
        onRetryLoad = { scope.launch { runCatching { LLMService.ensureLoaded(context) } } },
    )

    if (showSettings) {
        SettingsScreen(
            onShowOnboarding = { showSettings = false; showOnboarding = true },
            onDismiss = { showSettings = false },
        )
    }

    if (showOnboarding) {
        OnboardingScreen(onDone = {
            Prefs.setHasSeenOnboarding(context, true)
            showOnboarding = false
        })
    }
}
