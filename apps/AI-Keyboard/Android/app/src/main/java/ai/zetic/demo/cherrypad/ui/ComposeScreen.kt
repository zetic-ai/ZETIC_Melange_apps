package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.AppModel
import ai.zetic.demo.cherrypad.llm.LLMService
import ai.zetic.demo.cherrypad.model.KeyboardTask
import ai.zetic.demo.cherrypad.ui.theme.CherryColors
import ai.zetic.demo.cherrypad.ui.theme.CherryDims
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun ComposeScreen(
    model: AppModel,
    phase: LLMService.Phase,
    onOpenSettings: () -> Unit,
    onRetryLoad: () -> Unit,
) {
    var showLanguagePicker by remember { mutableStateOf(false) }
    val focus = LocalFocusManager.current

    Box(Modifier.fillMaxSize().background(CherryColors.Background)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp)
                .padding(bottom = 40.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Spacer(Modifier.size(28.dp))
                Spacer(Modifier.weight(1f))
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text("🍒", fontSize = 22.sp)
                    Text("CherryPad", fontSize = 22.sp, fontWeight = FontWeight.Black, color = CherryColors.TextPrimary)
                }
                Spacer(Modifier.weight(1f))
                Icon(
                    Icons.Filled.Settings,
                    contentDescription = "Settings",
                    tint = CherryColors.TextSecondary,
                    modifier = Modifier.size(28.dp).clickable { onOpenSettings() }.padding(4.dp),
                )
            }

            ModelStatusBanner(phase = phase, onRetry = onRetryLoad)

            // Input card
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .shadow(10.dp, RoundedCornerShape(CherryDims.cardRadius), clip = false)
                    .clip(RoundedCornerShape(CherryDims.cardRadius))
                    .background(CherryColors.Surface)
                    .padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Box(Modifier.fillMaxWidth().heightIn(min = 120.dp, max = 200.dp)) {
                    if (model.inputText.isEmpty()) {
                        Text(
                            "Type or paste your message…",
                            color = CherryColors.TextSecondary,
                            fontSize = 16.sp,
                            modifier = Modifier.padding(top = 8.dp, start = 2.dp),
                        )
                    }
                    BasicTextField(
                        value = model.inputText,
                        onValueChange = { model.inputText = it },
                        textStyle = TextStyle(color = CherryColors.TextPrimary, fontSize = 16.sp),
                        cursorBrush = SolidColor(CherryColors.Cherry),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                Text(
                    "${model.inputText.length}",
                    color = CherryColors.TextSecondary,
                    fontSize = 12.sp,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.End,
                )
            }

            ActionBar(selected = model.task, onSelect = { model.task = it })

            if (model.task != KeyboardTask.GRAMMAR) {
                ChipRow(model = model, onOpenLanguagePicker = { showLanguagePicker = true })
            }

            // Generate button
            val enabled = model.canGenerate
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(CherryDims.chipRadius))
                    .background(if (enabled) CherryColors.Cherry else CherryColors.Cherry.copy(alpha = 0.4f))
                    .clickable(enabled = enabled) { focus.clearFocus(); model.run() }
                    .padding(vertical = 15.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
            ) {
                Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = CherryColors.OnCherry, modifier = Modifier.size(18.dp))
                Spacer(Modifier.size(7.dp))
                Text(model.task.title, color = CherryColors.OnCherry, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }

            if (model.hasResult) {
                ResultCard(model = model)
            }
        }
    }

    if (showLanguagePicker) {
        LanguagePickerScreen(
            selected = model.targetLanguage,
            onSelect = { model.selectTargetLanguage(it); showLanguagePicker = false },
            onDismiss = { showLanguagePicker = false },
        )
    }
}
