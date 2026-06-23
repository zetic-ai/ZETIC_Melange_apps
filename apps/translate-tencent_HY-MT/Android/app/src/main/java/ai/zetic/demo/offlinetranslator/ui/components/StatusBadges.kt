package ai.zetic.demo.offlinetranslator.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import ai.zetic.demo.offlinetranslator.theme.Theme

/**
 * Live network status pill — reflects the *real* connection. Online is incidental; the demo's
 * point is that translation still works when this reads "Offline". Mirrors iOS `LiveStatusBadge`.
 */
@Composable
fun LiveStatusBadge(isOnline: Boolean, modifier: Modifier = Modifier) {
    val tint = if (isOnline) Theme.online else Theme.textSecondary
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(percent = 50))
            .background(tint.copy(alpha = 0.14f))
            .padding(horizontal = 8.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(tint))
        Spacer(Modifier.width(5.dp))
        Text(
            text = if (isOnline) "Online" else "Offline",
            color = tint,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold,
        )
    }
}

/** Tasteful "Powered by Zetic" footnote. Mirrors iOS `PoweredByZetic`. */
@Composable
fun PoweredByZetic(modifier: Modifier = Modifier) {
    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        Text(
            text = "On-device translation · powered by ",
            color = Theme.textTertiary,
            fontSize = 10.sp,
        )
        Text(
            text = "Zetic",
            color = Theme.textSecondary,
            fontSize = 10.sp,
            fontWeight = FontWeight.SemiBold,
        )
    }
}
