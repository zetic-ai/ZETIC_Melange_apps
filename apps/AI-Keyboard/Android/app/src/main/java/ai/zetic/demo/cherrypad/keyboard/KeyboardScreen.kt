package ai.zetic.demo.cherrypad.keyboard

import ai.zetic.demo.cherrypad.model.KeyboardTask
import ai.zetic.demo.cherrypad.ui.icon
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Backspace
import androidx.compose.material.icons.filled.ArrowDownward
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// Special key markers.
private const val SHIFT = "⇧"
private const val DELETE = "⌫"
private const val TO_SYMBOLS = "#+="
private const val TO_NUMBERS = "123"

private val letterRows = listOf(
    "q w e r t y u i o p".split(" "),
    "a s d f g h j k l".split(" "),
    listOf(SHIFT) + "z x c v b n m".split(" ") + listOf(DELETE),
)
private val numberRows = listOf(
    "1 2 3 4 5 6 7 8 9 0".split(" "),
    listOf("-", "/", ":", ";", "(", ")", "$", "&", "@", "\""),
    listOf(TO_SYMBOLS, ".", ",", "?", "!", "'", DELETE),
)
private val symbolRows = listOf(
    listOf("[", "]", "{", "}", "#", "%", "^", "*", "+", "="),
    listOf("_", "\\", "|", "~", "<", ">", "€", "£", "¥", "•"),
    listOf(TO_NUMBERS, ".", ",", "?", "!", "'", DELETE),
)

@Composable
fun KeyboardScreen(state: KeyboardState, actions: KeyboardActions) {
    val isPanel = state.processing || state.resultText != null
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .height(KB.keyboardHeight)
            .background(KB.background)
            .padding(horizontal = KB.sideMargin)
            .padding(top = KB.sectionGap, bottom = 4.dp),
        verticalArrangement = Arrangement.spacedBy(KB.sectionGap),
    ) {
        ActionBar(state, actions, if (isPanel) Modifier.fillMaxWidth().weight(1f) else Modifier.fillMaxWidth())
        if (!isPanel) KeysSection(state, actions, Modifier.fillMaxWidth().weight(1f))
    }
}

// ---- Action bar (idle / processing / result) ----

@Composable
private fun ActionBar(state: KeyboardState, actions: KeyboardActions, modifier: Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(KB.barGap)) {
        state.banner?.let { BannerCapsule(it) }
        when {
            state.processing -> ProcessingRow(state, Modifier.fillMaxWidth().weight(1f))
            state.resultText != null -> ResultPanel(state, actions, Modifier.fillMaxWidth().weight(1f))
            else -> ActionsRow(state, actions)
        }
    }
}

@Composable
private fun BannerCapsule(text: String) {
    Text(
        text = text,
        color = KB.cherryDark,
        fontSize = 12.sp,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(50))
            .background(KB.cherrySoft)
            .padding(vertical = 5.dp),
        maxLines = 2,
    )
}

@Composable
private fun ActionsRow(state: KeyboardState, actions: KeyboardActions) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(KB.barGap)) {
        for (t in KeyboardTask.entries) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .height(KB.actionButtonH)
                    .shadow(1.dp, RoundedCornerShape(KB.radiusControl), clip = false)
                    .clip(RoundedCornerShape(KB.radiusControl))
                    .background(KB.keyFill)
                    .clickable { state.banner = null; actions.runAction(t) },
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
            ) {
                Icon(t.icon, contentDescription = t.title, tint = KB.cherry, modifier = Modifier.size(19.dp))
                Spacer(Modifier.size(3.dp))
                Text(t.title, color = KB.cherry, fontSize = 11.sp, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}

@Composable
private fun ProcessingRow(state: KeyboardState, modifier: Modifier) {
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally), verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = KB.cherry)
        Text(state.statusText ?: "Thinking…", color = KB.textSecondary, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun ResultPanel(state: KeyboardState, actions: KeyboardActions, modifier: Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(KB.barGap)) {
        ResultCard(state.resultText ?: "")
        Spacer(Modifier.weight(1f))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(KB.barGap)) {
            SecondaryButton(Icons.Filled.Close, KB.textSecondary) { actions.dismissResult() }
            if (state.activeTask != null) {
                SecondaryButton(Icons.Filled.Refresh, KB.textPrimary) { actions.runAction(state.activeTask!!) }
            }
            Row(
                modifier = Modifier
                    .weight(1f)
                    .height(KB.controlHeight)
                    .clip(RoundedCornerShape(KB.radiusControl))
                    .background(KB.cherry)
                    .clickable { actions.insertResult() },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Filled.ArrowDownward, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                Spacer(Modifier.size(6.dp))
                Text("Insert result", color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun ColumnScope.ResultCard(text: String) {
    val inner: @Composable () -> Unit = {
        Text(text, color = KB.textPrimary, fontSize = 15.sp, modifier = Modifier.fillMaxWidth())
    }
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(KB.radiusControl))
            .background(KB.keyFill)
            .padding(horizontal = 12.dp, vertical = 10.dp),
    ) {
        if (text.length > 140) {
            Box(Modifier.heightIn(max = KB.resultMaxHeight).verticalScroll(rememberScrollState())) { inner() }
        } else {
            inner()
        }
    }
}

@Composable
private fun RowScope.SecondaryButton(icon: ImageVector, tint: Color, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .width(KB.secondaryWidth)
            .height(KB.controlHeight)
            .clip(RoundedCornerShape(KB.radiusControl))
            .background(KB.specialFill)
            .clickable { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = null, tint = tint, modifier = Modifier.size(18.dp))
    }
}

// ---- QWERTY ----

@Composable
private fun KeysSection(state: KeyboardState, actions: KeyboardActions, modifier: Modifier) {
    val rows = when (state.plane) {
        KeyPlane.LETTERS -> letterRows
        KeyPlane.NUMBERS -> numberRows
        KeyPlane.SYMBOLS -> symbolRows
    }
    Column(modifier, verticalArrangement = Arrangement.spacedBy(KB.keyGapV)) {
        for (row in rows) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(KB.keyGapH)) {
                for (key in row) KeyForToken(this, state, actions, key)
            }
        }
        BottomRow(state, actions)
    }
}

@Composable
private fun KeyForToken(scope: RowScope, state: KeyboardState, actions: KeyboardActions, key: String) = with(scope) {
    when (key) {
        SHIFT -> KeyCap(
            Modifier.width(KB.shiftDeleteWidth),
            icon = Icons.Filled.ArrowUpward,
            fill = if (state.shifted) KB.keyFill else KB.specialFill,
        ) { state.shifted = !state.shifted }
        DELETE -> KeyCap(
            Modifier.width(KB.shiftDeleteWidth),
            icon = Icons.AutoMirrored.Filled.Backspace,
            fill = KB.specialFill,
        ) { actions.deleteBackward() }
        TO_SYMBOLS -> KeyCap(Modifier.width(KB.inlinePlaneWidth), label = "#+=", fill = KB.specialFill, fontSize = 15.sp) { state.plane = KeyPlane.SYMBOLS }
        TO_NUMBERS -> KeyCap(Modifier.width(KB.inlinePlaneWidth), label = "123", fill = KB.specialFill, fontSize = 15.sp) { state.plane = KeyPlane.NUMBERS }
        else -> {
            val display = if (state.shifted) key.uppercase() else key
            KeyCap(Modifier.weight(1f), label = display, fill = KB.keyFill) { actions.insert(display) }
        }
    }
}

@Composable
private fun BottomRow(state: KeyboardState, actions: KeyboardActions) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(KB.keyGapH)) {
        val planeLabel = if (state.plane == KeyPlane.LETTERS) "123" else "ABC"
        KeyCap(Modifier.width(KB.planeToggleWidth), label = planeLabel, fill = KB.specialFill, fontSize = 15.sp) {
            state.plane = if (state.plane == KeyPlane.LETTERS) KeyPlane.NUMBERS else KeyPlane.LETTERS
        }
        KeyCap(Modifier.width(KB.globeWidth), icon = Icons.Filled.Language, fill = KB.specialFill) { actions.nextKeyboard() }
        KeyCap(Modifier.weight(1f), label = "space", fill = KB.keyFill, fontSize = 15.sp) { actions.insert(" ") }
        KeyCap(Modifier.width(KB.returnWidth), label = "return", fill = KB.specialFill, fontSize = 15.sp) { actions.newLine() }
    }
}

@Composable
private fun KeyCap(
    modifier: Modifier,
    label: String? = null,
    icon: ImageVector? = null,
    fill: Color,
    fontSize: androidx.compose.ui.unit.TextUnit = 20.sp,
    onClick: () -> Unit,
) {
    Box(
        modifier = modifier
            .height(KB.keyHeight)
            .shadow(1.dp, RoundedCornerShape(KB.radiusKey), clip = false)
            .clip(RoundedCornerShape(KB.radiusKey))
            .background(fill)
            .clickable { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = label, tint = KB.textPrimary, modifier = Modifier.size(20.dp))
        } else if (label != null) {
            Text(label, color = KB.textPrimary, fontSize = fontSize, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}
