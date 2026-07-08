package ai.zetic.demo.posemotion.ui

import ai.zetic.demo.posemotion.state.DemoViewModel
import ai.zetic.demo.posemotion.ui.overlay.Skeleton3dCanvas
import ai.zetic.demo.posemotion.ui.overlay.VideoFitMapper
import ai.zetic.demo.posemotion.ui.overlay.drawBallTrail
import ai.zetic.demo.posemotion.ui.overlay.drawBoundingBoxes
import ai.zetic.demo.posemotion.ui.overlay.drawSkeleton
import ai.zetic.demo.posemotion.video.ClipFrameSource
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ViewInAr
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.foundation.clickable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

@Composable
fun DemoScreen(viewModel: DemoViewModel) {
    val frame by viewModel.frame.collectAsState()
    val stats by viewModel.stats.collectAsState()
    val mode by viewModel.mode.collectAsState()
    val show3D by viewModel.show3D.collectAsState()
    val ballTrail by viewModel.ballTrail.collectAsState()

    Box(modifier = Modifier.fillMaxSize().background(Theme.Background)) {
        val f = frame
        if (f == null) {
            CircularProgressIndicator(color = Theme.Accent, modifier = Modifier.align(Alignment.Center))
        } else {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val fit = VideoFitMapper.fitRect(f.frameWidth.toFloat(), f.frameHeight.toFloat(), size)
                drawImage(
                    image = f.bitmap.asImageBitmap(),
                    dstOffset = IntOffset(fit.left.roundToInt(), fit.top.roundToInt()),
                    dstSize = IntSize(fit.width.roundToInt(), fit.height.roundToInt()),
                )
                drawBoundingBoxes(f.personBox, f.ballBox, fit)
                drawBallTrail(ballTrail, fit)
                f.keypoints?.let { drawSkeleton(it, fit) }
            }
        }

        // Video stays edge-to-edge; controls respect status bar / cutout / gesture nav.
        Column(modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(12.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
                BenchmarkHud(stats, viewModel.liftAvailable, mode)
                androidx.compose.foundation.layout.Spacer(Modifier.weight(1f))
                ClipPicker(viewModel)
            }

            androidx.compose.foundation.layout.Spacer(Modifier.weight(1f))

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Bottom,
            ) {
                ModePicker(mode) { viewModel.setMode(it) }
                androidx.compose.foundation.layout.Spacer(Modifier.weight(1f))
                if (viewModel.liftAvailable) {
                    Pose3dPanel(viewModel, show3D)
                }
            }
        }
    }
}

/** One numbered chip per bundled clip (shown only when there is a choice). */
@Composable
private fun ClipPicker(viewModel: DemoViewModel) {
    val clips by viewModel.availableClips.collectAsState()
    val selected by viewModel.selectedClip.collectAsState()
    if (clips.size < 2) return

    Column(
        modifier = Modifier
            .background(Theme.Card.copy(alpha = 0.82f), RoundedCornerShape(18.dp))
            .padding(3.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        clips.forEachIndexed { index, name ->
            val active = name == selected
            Text(
                "${index + 1}",
                color = if (active) Color.Black else Theme.TextSecondary,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .padding(vertical = 2.dp)
                    .background(if (active) Theme.Accent else Color.Transparent, CircleShape)
                    .clickable { viewModel.setClip(name) }
                    .padding(horizontal = 11.dp, vertical = 6.dp),
            )
        }
    }
}

@Composable
private fun ModePicker(selected: ClipFrameSource.Mode, onSelect: (ClipFrameSource.Mode) -> Unit) {
    Row(
        modifier = Modifier
            .background(Theme.Card.copy(alpha = 0.82f), CircleShape)
            .padding(3.dp),
    ) {
        ClipFrameSource.Mode.entries.forEach { m ->
            val active = m == selected
            Text(
                m.label,
                color = if (active) Color.Black else Theme.TextSecondary,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier
                    .background(if (active) Theme.Accent else Color.Transparent, CircleShape)
                    .clickable { onSelect(m) }
                    .padding(horizontal = 14.dp, vertical = 8.dp),
            )
        }
    }
}

@Composable
private fun Pose3dPanel(viewModel: DemoViewModel, show3D: Boolean) {
    val frame by viewModel.frame.collectAsState()
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        TextButton(onClick = { viewModel.toggle3D() }) {
            Icon(
                Icons.Default.ViewInAr, contentDescription = "3D",
                tint = if (show3D) Theme.Accent else Theme.TextSecondary,
                modifier = Modifier.size(14.dp),
            )
            Text(
                " 3D",
                color = if (show3D) Theme.Accent else Theme.TextSecondary,
                fontSize = 11.sp, fontWeight = FontWeight.Bold,
            )
        }
        if (show3D) {
            Box(
                modifier = Modifier
                    .size(width = 150.dp, height = 170.dp)
                    .background(Theme.Card.copy(alpha = 0.82f), RoundedCornerShape(14.dp)),
                contentAlignment = Alignment.Center,
            ) {
                val joints = frame?.pose3D
                if (joints != null) {
                    Skeleton3dCanvas(joints, Modifier.fillMaxSize())
                } else {
                    Text("Gathering motion…", color = Theme.TextSecondary, fontSize = 11.sp)
                }
            }
        }
    }
}
