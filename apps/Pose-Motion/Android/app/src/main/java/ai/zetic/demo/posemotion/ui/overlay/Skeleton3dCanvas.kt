package ai.zetic.demo.posemotion.ui.overlay

import ai.zetic.demo.posemotion.pipeline.Vec3
import ai.zetic.demo.posemotion.ui.Theme
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.input.pointer.pointerInput
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Side-view 3D skeleton: orthographic projection of the root-relative H36M joints,
 * rotated about the vertical axis by a drag-controlled azimuth.
 */
@Composable
fun Skeleton3dCanvas(joints: List<Vec3>, modifier: Modifier = Modifier) {
    var azimuth by remember { mutableFloatStateOf(0.9f) }   // radians; start slightly rotated

    Canvas(
        modifier = modifier.pointerInput(Unit) {
            detectDragGestures { change, dragAmount ->
                change.consume()
                azimuth += dragAmount.x * 0.012f
            }
        }
    ) {
        if (joints.size != 17) return@Canvas
        val cosA = cos(azimuth)
        val sinA = sin(azimuth)

        // Rotate about Y (vertical); keep image-style y-down.
        val rotated = joints.map {
            Vec3(it.x * cosA + it.z * sinA, it.y, -it.x * sinA + it.z * cosA)
        }

        val maxExtent = rotated.fold(1e-4f) { m, p -> maxOf(m, abs(p.x), abs(p.y)) }
        val scale = 0.42f * minOf(size.width, size.height) / maxExtent
        val center = Offset(size.width / 2, size.height / 2)
        fun project(p: Vec3) = Offset(center.x + p.x * scale, center.y + p.y * scale)

        val zMin = rotated.minOf { it.z }
        val zMax = rotated.maxOf { it.z }
        fun depthAlpha(z: Float): Float {
            val t = (z - zMin) / maxOf(1e-4f, zMax - zMin)
            return 0.35f + 0.65f * (1f - t)
        }

        // Depth-sorted bones, nearer drawn brighter and on top.
        val sortedBones = BONES.sortedByDescending { rotated[it.first].z + rotated[it.second].z }
        for ((a, b) in sortedBones) {
            val color: Color = when {
                b in LEFT_JOINTS -> Theme.LeftSide
                b in RIGHT_JOINTS -> Theme.RightSide
                else -> Theme.Torso
            }
            val alpha = depthAlpha((rotated[a].z + rotated[b].z) / 2)
            drawLine(
                color.copy(alpha = alpha),
                project(rotated[a]), project(rotated[b]),
                strokeWidth = 3f, cap = StrokeCap.Round,
            )
        }
        for (p in rotated) {
            drawCircle(Theme.TextPrimary.copy(alpha = depthAlpha(p.z)), 2.5f, project(p))
        }
    }
}

private val BONES = listOf(
    0 to 1, 1 to 2, 2 to 3,           // right leg
    0 to 4, 4 to 5, 5 to 6,           // left leg
    0 to 7, 7 to 8, 8 to 9, 9 to 10,  // spine → head
    8 to 11, 11 to 12, 12 to 13,      // left arm
    8 to 14, 14 to 15, 15 to 16,      // right arm
)
private val LEFT_JOINTS = setOf(4, 5, 6, 11, 12, 13)
private val RIGHT_JOINTS = setOf(1, 2, 3, 14, 15, 16)
