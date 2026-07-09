package ai.zetic.demo.posemotion.ui

import ai.zetic.demo.posemotion.pipeline.BenchmarkSnapshot
import ai.zetic.demo.posemotion.video.ClipFrameSource
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.util.Locale

/** The demo's selling point: live per-model latency, sustained FPS, and peak memory. */
@Composable
fun BenchmarkHud(
    stats: BenchmarkSnapshot,
    liftAvailable: Boolean,
    mode: ClipFrameSource.Mode,
) {
    Column(
        modifier = Modifier
            .width(220.dp)
            .background(Theme.Card.copy(alpha = 0.82f), RoundedCornerShape(14.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                Icons.Default.Memory, contentDescription = null,
                tint = Theme.Accent, modifier = Modifier.size(13.dp),
            )
            Text(
                " On-device · Melange",
                color = Theme.Accent, fontSize = 11.sp, fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.weight(1f))
            Text(
                if (mode == ClipFrameSource.Mode.BENCHMARK) "sustained" else "realtime",
                color = Theme.TextSecondary, fontSize = 10.sp, fontWeight = FontWeight.SemiBold,
            )
        }

        Row(modifier = Modifier.fillMaxWidth()) {
            Text(
                "memory",
                color = Theme.TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.Medium,
            )
            Spacer(Modifier.weight(1f))
            Text(
                String.format(Locale.US, "%.0f MB · peak %.0f", stats.memoryMB, stats.peakMemoryMB),
                color = Theme.TextPrimary, fontSize = 11.sp, fontWeight = FontWeight.Medium,
            )
        }

        LatencyRow("YOLO26n", stats.detectMs)
        LatencyRow("RTMPose-s", stats.poseMs)
        if (liftAvailable) LatencyRow("3D lift", stats.liftMs)
        LatencyRow("pipeline", stats.totalMs, emphasized = true)
    }
}

@Composable
private fun LatencyRow(label: String, ms: Double, emphasized: Boolean = false) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Text(
            label,
            color = if (emphasized) Theme.TextPrimary else Theme.TextSecondary,
            fontSize = 11.sp,
            fontWeight = if (emphasized) FontWeight.Bold else FontWeight.Medium,
        )
        Spacer(Modifier.weight(1f))
        Text(
            if (ms > 0) String.format(Locale.US, "%.1f ms", ms) else "—",
            color = if (emphasized) Theme.Accent else Theme.TextPrimary,
            fontSize = 11.sp,
            fontWeight = if (emphasized) FontWeight.Bold else FontWeight.Medium,
        )
    }
}
