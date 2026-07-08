package ai.zetic.demo.posemotion.ui

import ai.zetic.demo.posemotion.state.DemoViewModel
import ai.zetic.demo.posemotion.state.ModelLoadState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.SportsGolf
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun ModelDownloadScreen(viewModel: DemoViewModel) {
    val loadStates by viewModel.loadStates.collectAsState()

    Column(
        modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(horizontal = 28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Default.SportsGolf, contentDescription = null,
            tint = Theme.Accent, modifier = Modifier.size(48.dp),
        )
        Text(
            "Pose & Motion",
            color = Theme.TextPrimary, fontSize = 24.sp, fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(top = 8.dp),
        )
        Text(
            "Preparing on-device models",
            color = Theme.TextSecondary, fontSize = 14.sp,
            modifier = Modifier.padding(top = 4.dp, bottom = 26.dp),
        )
        loadStates.forEachIndexed { index, state ->
            ModelLoadRow(state) { viewModel.retry(index) }
        }
    }
}

@Composable
private fun ModelLoadRow(state: ModelLoadState, retry: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .background(Theme.Card, RoundedCornerShape(18.dp))
            .padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(modifier = Modifier.size(34.dp), contentAlignment = Alignment.Center) {
            when {
                state.loaded -> Icon(
                    Icons.Default.Check, contentDescription = "ready",
                    tint = Theme.Good, modifier = Modifier.size(20.dp),
                )
                state.error != null -> Icon(
                    Icons.Default.Warning, contentDescription = "error",
                    tint = Theme.Poor, modifier = Modifier.size(20.dp),
                )
                state.progress > 0f -> CircularProgressIndicator(
                    progress = { state.progress },
                    color = Theme.Accent, trackColor = Theme.AccentSoft,
                    strokeWidth = 4.dp, modifier = Modifier.size(34.dp),
                )
                else -> CircularProgressIndicator(
                    color = Theme.Accent, trackColor = Theme.AccentSoft,
                    strokeWidth = 4.dp, modifier = Modifier.size(34.dp),
                )
            }
        }

        Column(modifier = Modifier.padding(start = 14.dp).weight(1f)) {
            Text(
                state.label,
                color = Theme.TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.SemiBold,
            )
            val subtitle = when {
                state.error != null && state.optional -> "Unavailable — demo continues in 2D"
                state.error != null -> state.error
                state.loaded -> "Ready"
                state.progress > 0f -> "Downloading ${(state.progress * 100).toInt()}%"
                else -> "Optimizing for this device…"
            }
            Text(
                subtitle,
                color = if (state.error != null && !state.optional) Theme.Poor else Theme.TextSecondary,
                fontSize = 12.sp, maxLines = 2,
            )
        }

        if (state.error != null) {
            Spacer(Modifier.width(6.dp))
            IconButton(onClick = retry) {
                Icon(Icons.Default.Refresh, contentDescription = "retry", tint = Theme.Accent)
            }
        }
    }
}
