package ai.zetic.demo.offlinetranslator.ui

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import ai.zetic.demo.offlinetranslator.model.Language
import ai.zetic.demo.offlinetranslator.theme.Theme
import ai.zetic.demo.offlinetranslator.ui.components.LanguageBar
import ai.zetic.demo.offlinetranslator.ui.components.LiveStatusBadge
import ai.zetic.demo.offlinetranslator.ui.components.PoweredByZetic
import ai.zetic.demo.offlinetranslator.viewmodel.ModelState
import ai.zetic.demo.offlinetranslator.viewmodel.TranslationViewModel
import kotlin.math.roundToInt

private enum class Phase { Idle, Editing, Result }
private enum class PickerField { Source, Target }

/**
 * Single state-driven screen mirroring DeepL's flow: idle → editing → result, with a persistent
 * language bar. Faithful port of iOS `TranslatorScreen`.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranslatorScreen(modifier: Modifier = Modifier, vm: TranslationViewModel = viewModel()) {
    val sourceText by vm.sourceText.collectAsStateWithLifecycle()
    val translatedText by vm.translatedText.collectAsStateWithLifecycle()
    val source by vm.source.collectAsStateWithLifecycle()
    val target by vm.target.collectAsStateWithLifecycle()
    val modelState by vm.modelState.collectAsStateWithLifecycle()
    val downloadProgress by vm.downloadProgress.collectAsStateWithLifecycle()
    val isTranslating by vm.isTranslating.collectAsStateWithLifecycle()
    val isOnline by vm.isOnline.collectAsStateWithLifecycle()
    val isListening by vm.isListening.collectAsStateWithLifecycle()
    val partialVoiceText by vm.partialVoiceText.collectAsStateWithLifecycle()
    val isRecognizingImage by vm.isRecognizingImage.collectAsStateWithLifecycle()
    val inputError by vm.inputError.collectAsStateWithLifecycle()

    val context = LocalContext.current
    var phase by remember { mutableStateOf(Phase.Idle) }
    var picker by remember { mutableStateOf<PickerField?>(null) }
    var showImageChooser by remember { mutableStateOf(false) }
    var pendingCameraUri by remember { mutableStateOf<Uri?>(null) }
    val focusRequester = remember { FocusRequester() }

    val ready = modelState == ModelState.Ready
    val canTranslate = ready && sourceText.trim().isNotEmpty()

    // Voice/OCR captured text → jump to the result screen (translation auto-starts in the VM).
    LaunchedEffect(Unit) { vm.showResult.collect { phase = Phase.Result } }

    // Image OCR launchers: camera capture (FileProvider Uri) and the Android Photo Picker.
    val takePicture = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        pendingCameraUri?.let { if (success) vm.recognizeImage(it) }
        pendingCameraUri = null
    }
    val pickMedia = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let { vm.recognizeImage(it) } }

    fun launchCamera() {
        val uri = createCaptureUri(context)
        pendingCameraUri = uri
        takePicture.launch(uri)
    }

    val micPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) vm.startVoiceInput() }
    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) launchCamera() }

    fun onVoice() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) {
            vm.startVoiceInput()
        } else {
            micPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    fun onCamera() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            launchCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    fun startEditing() {
        phase = Phase.Editing
    }

    fun confirmTranslate() {
        vm.translate()
        phase = Phase.Result
    }

    fun retranslateIfNeeded() {
        if (phase == Phase.Result) vm.translate()
    }

    fun share(text: String) {
        if (text.isEmpty()) return
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, text)
        }
        context.startActivity(Intent.createChooser(intent, null))
    }

    // Focus the editor whenever we enter the editing phase.
    LaunchedEffect(phase) {
        if (phase == Phase.Editing) {
            runCatching { focusRequester.requestFocus() }
        }
    }

    Box(modifier) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                // Edge-to-edge: keep content clear of the status bar, camera cutout, gesture nav
                // bar, and the IME so nothing is clipped or hidden on devices like the Galaxy S24.
                .safeDrawingPadding()
                .padding(top = 8.dp),
            verticalArrangement = Arrangement.spacedBy(11.dp),
        ) {
            // Header
            Box(Modifier.fillMaxWidth().padding(horizontal = 17.dp)) {
                when (phase) {
                    Phase.Idle -> IdleHeader(isOnline)
                    Phase.Editing -> EditingHeader(
                        isOnline = isOnline,
                        canTranslate = canTranslate,
                        onBack = { phase = Phase.Idle },
                        onConfirm = { confirmTranslate() },
                    )
                    Phase.Result -> ResultHeader(
                        isOnline = isOnline,
                        onBack = { phase = Phase.Editing },
                        onClose = { vm.clearAll(); phase = Phase.Idle },
                    )
                }
            }

            // Card with the loading overlay on top.
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(horizontal = 14.dp)
                    .clip(RoundedCornerShape(22.dp))
                    .background(Theme.surface),
            ) {
                Box(Modifier.fillMaxSize().padding(20.dp)) {
                    when (phase) {
                        Phase.Idle -> IdleView(
                            modelReady = ready,
                            onActivate = { startEditing() },
                            onPaste = { vm.paste() },
                            onVoice = { onVoice() },
                            onImage = { showImageChooser = true },
                        )
                        Phase.Editing -> EditingView(
                            sourceText = sourceText,
                            onSourceTextChange = { vm.setSourceText(it) },
                            modelReady = ready,
                            onPaste = { vm.paste() },
                            focusRequester = focusRequester,
                        )
                        Phase.Result -> ResultView(
                            sourceText = sourceText,
                            translatedText = translatedText,
                            isTranslating = isTranslating,
                            onEditSource = { phase = Phase.Editing },
                            onSpeakSource = { vm.speak(sourceText, source) },
                            onSpeakTranslation = { vm.speak(translatedText, target) },
                            onShare = { share(translatedText) },
                            onCopy = { vm.copyTranslation() },
                        )
                    }
                }
                ModelLoadingOverlay(modelState, downloadProgress, onRetry = { vm.loadModel() })
                if (isListening) {
                    ListeningOverlay(partialText = partialVoiceText, onStop = { vm.stopVoiceInput() })
                } else if (isRecognizingImage) {
                    BusyOverlay("Reading text from image…")
                }
            }

            // Language bar
            LanguageBar(
                source = source,
                target = target,
                onTapSource = { picker = PickerField.Source },
                onTapTarget = { picker = PickerField.Target },
                onSwap = { vm.swapLanguages() },
                modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp),
            )

            Box(Modifier.fillMaxWidth().padding(bottom = 4.dp), contentAlignment = Alignment.Center) {
                PoweredByZetic()
            }
        }
    }

    // Language picker sheet
    picker?.let { field ->
        val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
        ModalBottomSheet(
            onDismissRequest = { picker = null },
            sheetState = sheetState,
            containerColor = Theme.background,
        ) {
            Box(Modifier.fillMaxHeight(0.9f)) {
                when (field) {
                    PickerField.Source -> LanguagePickerView(
                        title = "Translate from",
                        options = Language.sourceOptions,
                        selected = source,
                        onSelect = { vm.setSource(it); picker = null; retranslateIfNeeded() },
                        onDismiss = { picker = null },
                    )
                    PickerField.Target -> LanguagePickerView(
                        title = "Translate to",
                        options = Language.targetOptions,
                        selected = target,
                        onSelect = { vm.setTarget(it); picker = null; retranslateIfNeeded() },
                        onDismiss = { picker = null },
                    )
                }
            }
        }
    }

    // Camera / Photo Library chooser for image OCR.
    if (showImageChooser) {
        AlertDialog(
            onDismissRequest = { showImageChooser = false },
            containerColor = Theme.surface,
            title = { Text("Translate from image", color = Theme.textPrimary) },
            text = { Text("Capture a photo or choose one from your library.", color = Theme.textSecondary) },
            confirmButton = {
                TextButton(onClick = { showImageChooser = false; onCamera() }) {
                    Icon(Icons.Filled.PhotoCamera, null, tint = Theme.accent, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Camera", color = Theme.accent)
                }
            },
            dismissButton = {
                TextButton(onClick = {
                    showImageChooser = false
                    pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                }) {
                    Icon(Icons.Filled.PhotoLibrary, null, tint = Theme.accent, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Library", color = Theme.accent)
                }
            },
        )
    }

    // Surface voice/OCR errors (permission denied, no text found, recognizer unavailable…).
    inputError?.let { message ->
        AlertDialog(
            onDismissRequest = { vm.clearInputError() },
            containerColor = Theme.surface,
            title = { Text("Input failed", color = Theme.textPrimary) },
            text = { Text(message, color = Theme.textSecondary) },
            confirmButton = {
                TextButton(onClick = { vm.clearInputError() }) { Text("OK", color = Theme.accent) }
            },
        )
    }
}

// MARK: - Headers

@Composable
private fun IdleHeader(isOnline: Boolean) {
    Box(Modifier.fillMaxWidth().height(40.dp), contentAlignment = Alignment.Center) {
        SegmentedControl()
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Filled.AccountCircle,
                contentDescription = "Profile",
                tint = Theme.accent,
                modifier = Modifier.size(30.dp),
            )
            Spacer(Modifier.weight(1f))
            LiveStatusBadge(isOnline)
        }
    }
}

@Composable
private fun EditingHeader(isOnline: Boolean, canTranslate: Boolean, onBack: () -> Unit, onConfirm: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().height(40.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        HeaderIcon(Icons.AutoMirrored.Filled.ArrowBack, onBack)
        Spacer(Modifier.weight(1f))
        LiveStatusBadge(isOnline)
        Spacer(Modifier.weight(1f))
        HeaderIcon(
            Icons.Filled.Check,
            onConfirm,
            tint = if (canTranslate) Theme.textPrimary else Theme.textTertiary,
            enabled = canTranslate,
        )
    }
}

@Composable
private fun ResultHeader(isOnline: Boolean, onBack: () -> Unit, onClose: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().height(40.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        HeaderIcon(Icons.AutoMirrored.Filled.ArrowBack, onBack)
        Spacer(Modifier.weight(1f))
        LiveStatusBadge(isOnline)
        Spacer(Modifier.weight(1f))
        HeaderIcon(Icons.Filled.Close, onClose)
    }
}

@Composable
private fun HeaderIcon(
    icon: ImageVector,
    onClick: () -> Unit,
    tint: Color = Theme.textPrimary,
    enabled: Boolean = true,
) {
    Box(
        modifier = Modifier
            .size(40.dp)
            .clickable(
                enabled = enabled,
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = onClick,
            ),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = null, tint = tint, modifier = Modifier.size(22.dp))
    }
}

@Composable
private fun SegmentedControl() {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(percent = 50))
            .background(Theme.surfaceRaised)
            .padding(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Segment("Translator", Icons.Filled.Translate, selected = true)
        Spacer(Modifier.width(4.dp))
        Segment(null, Icons.Filled.AutoFixHigh, selected = false)
    }
}

@Composable
private fun Segment(title: String?, icon: ImageVector, selected: Boolean) {
    val content = if (selected) Color.White else Theme.textSecondary
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(percent = 50))
            .background(if (selected) Theme.accentDeep else Color.Transparent)
            .padding(horizontal = 13.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = content, modifier = Modifier.size(14.dp))
        if (title != null) {
            Spacer(Modifier.width(5.dp))
            Text(title, color = content, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

// MARK: - Model loading overlay

@Composable
private fun ModelLoadingOverlay(state: ModelState, progress: Double, onRetry: () -> Unit) {
    when (state) {
        ModelState.Ready -> Unit
        ModelState.Loading -> LoadingCard {
            CircularProgressIndicator(color = Theme.accent)
            val title = if (progress > 0.0 && progress < 1.0) {
                "Downloading model… ${(progress * 100).roundToInt()}%"
            } else {
                "Preparing model…"
            }
            Text(title, color = Theme.textPrimary, fontSize = 17.sp, fontWeight = FontWeight.SemiBold)
            if (progress > 0.0 && progress < 1.0) {
                LinearProgressIndicator(
                    progress = { progress.toFloat() },
                    color = Theme.accent,
                    modifier = Modifier.widthIn(max = 220.dp).fillMaxWidth(),
                )
            }
            Text(
                "Preparing on-device translation. The model downloads once, then works fully offline.",
                color = Theme.textSecondary,
                fontSize = 13.sp,
                textAlign = TextAlign.Center,
            )
        }
        is ModelState.Failed -> LoadingCard {
            Icon(Icons.Filled.Warning, contentDescription = null, tint = Color(0xFFFF9F0A), modifier = Modifier.size(26.dp))
            Text("Couldn't load the model", color = Theme.textPrimary, fontSize = 17.sp, fontWeight = FontWeight.SemiBold)
            Text(state.message, color = Theme.textSecondary, fontSize = 13.sp, textAlign = TextAlign.Center)
            Box(
                modifier = Modifier
                    .padding(top = 4.dp)
                    .clip(RoundedCornerShape(percent = 50))
                    .background(Theme.accent)
                    .clickable(onClick = onRetry)
                    .padding(horizontal = 24.dp, vertical = 10.dp),
            ) {
                Text("Retry", color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
private fun LoadingCard(content: @Composable androidx.compose.foundation.layout.ColumnScope.() -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .clip(RoundedCornerShape(22.dp))
            .background(Theme.surface),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier.padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp),
            content = content,
        )
    }
}

// MARK: - Voice / OCR overlays

@Composable
private fun ListeningOverlay(partialText: String, onStop: () -> Unit) {
    LoadingCard {
        Icon(Icons.Filled.Mic, contentDescription = null, tint = Theme.accent, modifier = Modifier.size(40.dp))
        Text("Listening…", color = Theme.textPrimary, fontSize = 17.sp, fontWeight = FontWeight.SemiBold)
        Text(
            text = partialText.ifEmpty { "Speak now, then tap Stop." },
            color = if (partialText.isEmpty()) Theme.textSecondary else Theme.textPrimary,
            fontSize = if (partialText.isEmpty()) 13.sp else 18.sp,
            textAlign = TextAlign.Center,
        )
        Box(
            modifier = Modifier
                .padding(top = 4.dp)
                .clip(RoundedCornerShape(percent = 50))
                .background(Theme.accent)
                .clickable(onClick = onStop)
                .padding(horizontal = 24.dp, vertical = 10.dp),
        ) {
            Text("Stop", color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun BusyOverlay(message: String) {
    LoadingCard {
        CircularProgressIndicator(color = Theme.accent)
        Text(message, color = Theme.textPrimary, fontSize = 15.sp, textAlign = TextAlign.Center)
    }
}

/** Creates a FileProvider Uri in cache/ocr/ for full-resolution camera capture. */
private fun createCaptureUri(context: Context): Uri {
    val dir = java.io.File(context.cacheDir, "ocr").apply { mkdirs() }
    val file = java.io.File.createTempFile("capture_", ".jpg", dir)
    return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
}
