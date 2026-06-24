package ai.zetic.demo.offlinetranslator.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.demo.offlinetranslator.theme.Theme
import ai.zetic.demo.offlinetranslator.ui.components.PasteButton

/**
 * Text-entry state: an editable field with placeholder and a Paste button while empty.
 * Mirrors iOS `EditingView`.
 */
@Composable
fun EditingView(
    sourceText: String,
    onSourceTextChange: (String) -> Unit,
    modelReady: Boolean,
    onPaste: () -> Unit,
    focusRequester: FocusRequester,
) {
    val textStyle = TextStyle(
        color = Theme.textPrimary,
        fontSize = 24.sp,
        fontWeight = FontWeight.Normal,
    )
    Column(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxWidth().heightIn(min = 130.dp)) {
            if (sourceText.isEmpty()) {
                Text(
                    text = "Type or paste here to translate",
                    color = Theme.textSecondary,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Normal,
                )
            }
            BasicTextField(
                value = sourceText,
                onValueChange = onSourceTextChange,
                modifier = Modifier.fillMaxWidth().focusRequester(focusRequester),
                textStyle = LocalTextStyle.current.merge(textStyle),
                cursorBrush = SolidColor(Theme.accent),
            )
        }

        if (sourceText.isEmpty()) {
            Box(Modifier.padding(top = 16.dp)) {
                PasteButton(enabled = modelReady, onClick = onPaste)
            }
        }
    }
}
