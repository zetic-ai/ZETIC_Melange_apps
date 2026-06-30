package com.brew

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.brew.engine.ModelPhase
import com.brew.ui.ChatSheet
import com.brew.ui.NoteDetailScreen
import com.brew.ui.NotesListScreen
import com.brew.ui.RecordingSheet
import com.brew.ui.SettingsSheet
import com.brew.ui.theme.BrewColors
import com.brew.vm.RecordingViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BrewApp() {
    val app = BrewApplication.instance
    val navController = rememberNavController()
    val recordingVm: RecordingViewModel = viewModel()
    val recording by recordingVm.state.collectAsStateWithLifecycle()
    val phase by app.coordinator.llmPhase.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()

    var showRecording by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }
    var chatNoteId by remember { mutableStateOf<String?>(null) }

    val aiReady = phase is ModelPhase.Ready || phase is ModelPhase.Failed

    // Recover interrupted transcriptions and pre-warm the model on first launch.
    LaunchedEffect(Unit) {
        app.transcriptionWorker.recoverInterruptedWork()
        scope.launch { app.coordinator.preloadLlm() }
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            recordingVm.start()
            showRecording = true
        }
    }

    NavHost(navController = navController, startDestination = "notes", modifier = Modifier.fillMaxSize()) {
        composable("notes") {
            NotesListScreen(
                recording = recording,
                onOpenNote = { id -> navController.navigate("note/$id") },
                onStartRecording = {
                    permissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                },
                onReopenRecording = { showRecording = true },
                onOpenSettings = { showSettings = true },
                aiReady = aiReady,
            )
        }
        composable("note/{id}") { entry ->
            val id = entry.arguments?.getString("id") ?: return@composable
            NoteDetailScreen(
                noteId = id,
                onBack = { navController.popBackStack() },
                onOpenChat = { chatNoteId = id },
            )
        }
    }

    if (showRecording) {
        ModalBottomSheet(
            onDismissRequest = { showRecording = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = BrewColors.canvas,
        ) {
            RecordingSheet(
                state = recording,
                onPause = recordingVm::pause,
                onResume = recordingVm::resume,
                onStop = {
                    recordingVm.stopAndSave { id ->
                        showRecording = false
                        navController.navigate("note/$id")
                    }
                },
                onCancel = {
                    recordingVm.cancel()
                    showRecording = false
                },
            )
        }
    }

    if (showSettings) {
        ModalBottomSheet(
            onDismissRequest = { showSettings = false },
            containerColor = BrewColors.canvas,
        ) { SettingsSheet() }
    }

    chatNoteId?.let { id ->
        ModalBottomSheet(
            onDismissRequest = { chatNoteId = null },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = BrewColors.canvas,
        ) { ChatSheet(noteId = id) }
    }
}
