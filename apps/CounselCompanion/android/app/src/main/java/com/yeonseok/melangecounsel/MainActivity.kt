package com.yeonseok.melangecounsel

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.CreationExtras
import com.yeonseok.melangecounsel.ui.AppRoot
import com.yeonseok.melangecounsel.ui.theme.CounselTheme
import com.yeonseok.melangecounsel.ui.theme.ThemeMode
import com.yeonseok.melangecounsel.ui.viewmodel.CounselViewModel

class MainActivity : ComponentActivity() {

    private val viewModel: CounselViewModel by lazy {
        ViewModelProvider(this, CounselViewModelFactory)[CounselViewModel::class.java]
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val state = viewModel.uiState.collectAsStateWithLifecycle().value
            CounselTheme(themeMode = ThemeMode.valueOf(state.themeMode)) {
                AppRoot(state = state, viewModel = viewModel)
            }
        }
    }
}

object CounselViewModelFactory : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
        return CounselViewModel(
            chatRepository = AppContainer.chatRepository,
            settingsRepository = AppContainer.settingsRepository,
            diagnosticsRepository = AppContainer.diagnosticsRepository,
            connectivityObserver = AppContainer.connectivityObserver,
            chatEngine = AppContainer.chatEngine
        ) as T
    }
}
