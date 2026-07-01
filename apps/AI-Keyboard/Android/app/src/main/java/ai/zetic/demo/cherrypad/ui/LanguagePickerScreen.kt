package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.model.Language
import ai.zetic.demo.cherrypad.ui.theme.CherryColors
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
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
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties

@Composable
fun LanguagePickerScreen(
    selected: Language,
    onSelect: (Language) -> Unit,
    onDismiss: () -> Unit,
) {
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        var query by remember { mutableStateOf("") }
        val filtered = Language.all.filter {
            query.isBlank() ||
                it.englishName.contains(query, ignoreCase = true) ||
                it.nativeName.contains(query, ignoreCase = true)
        }
        Column(Modifier.fillMaxSize().background(CherryColors.Background)) {
            // Top bar
            Row(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Translate to", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = CherryColors.TextPrimary)
                Spacer(Modifier.weight(1f))
                Text("Done", color = CherryColors.Cherry, fontSize = 16.sp, fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.clickable { onDismiss() })
            }
            // Search
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(CherryColors.SurfaceMuted)
                    .padding(horizontal = 12.dp, vertical = 10.dp),
            ) {
                if (query.isEmpty()) {
                    Text("Search languages", color = CherryColors.TextSecondary, fontSize = 15.sp)
                }
                BasicTextField(
                    value = query,
                    onValueChange = { query = it },
                    singleLine = true,
                    textStyle = TextStyle(color = CherryColors.TextPrimary, fontSize = 15.sp),
                    cursorBrush = SolidColor(CherryColors.Cherry),
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            LazyColumn(Modifier.fillMaxWidth().padding(top = 8.dp)) {
                items(filtered) { lang ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onSelect(lang) }
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(lang.englishName, fontSize = 16.sp, fontWeight = FontWeight.Medium, color = CherryColors.TextPrimary)
                            if (lang.nativeName != lang.englishName) {
                                Text(lang.nativeName, fontSize = 12.sp, color = CherryColors.TextSecondary)
                            }
                        }
                        if (lang.id == selected.id) {
                            Icon(Icons.Filled.Check, contentDescription = "Selected", tint = CherryColors.Cherry, modifier = Modifier.size(18.dp))
                        }
                    }
                }
            }
        }
    }
}
