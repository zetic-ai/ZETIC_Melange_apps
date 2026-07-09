package ai.zetic.demo.posemotion.ui.overlay

import ai.zetic.demo.posemotion.AppConfig
import ai.zetic.demo.posemotion.pipeline.Keypoint2D
import ai.zetic.demo.posemotion.ui.Theme
import android.graphics.PointF
import android.graphics.RectF
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke

/** Person + ball boxes over the video. */
fun DrawScope.drawBoundingBoxes(personBox: RectF?, ballBox: RectF?, fit: Rect) {
    personBox?.let { drawBox(it, Theme.Accent, fit) }
    ballBox?.let { drawBox(it, Theme.Ball, fit) }
}

private fun DrawScope.drawBox(box: RectF, color: Color, fit: Rect) {
    val r = VideoFitMapper.rect(box, fit)
    drawRoundRect(
        color = color.copy(alpha = 0.9f),
        topLeft = Offset(r.left, r.top),
        size = r.size,
        cornerRadius = CornerRadius(6f, 6f),
        style = Stroke(width = 2f),
    )
}

/** Fading trail through the last N ball centers, with a bright head dot. */
fun DrawScope.drawBallTrail(trail: List<PointF>, fit: Rect) {
    if (trail.size < 2) return
    val points = trail.map { VideoFitMapper.point(it.x, it.y, fit) }
    for (i in 1 until points.size) {
        val alpha = i.toFloat() / points.size
        drawLine(
            color = Theme.Ball.copy(alpha = 0.15f + 0.75f * alpha),
            start = points[i - 1],
            end = points[i],
            strokeWidth = 2f + 2f * alpha,
            cap = StrokeCap.Round,
        )
    }
    val head = points.last()
    drawCircle(Theme.Ball, radius = 6f, center = head)
    drawCircle(
        Theme.Ball.copy(alpha = 0.5f), radius = 9f, center = head,
        style = Stroke(width = 2f),
    )
}

// (a, b, left/right/torso) — COCO indices
private val bones: List<Triple<Int, Int, Color>> = listOf(
    Triple(5, 7, Theme.LeftSide), Triple(7, 9, Theme.LeftSide),          // left arm
    Triple(6, 8, Theme.RightSide), Triple(8, 10, Theme.RightSide),       // right arm
    Triple(11, 13, Theme.LeftSide), Triple(13, 15, Theme.LeftSide),      // left leg
    Triple(12, 14, Theme.RightSide), Triple(14, 16, Theme.RightSide),    // right leg
    Triple(5, 6, Theme.Torso), Triple(11, 12, Theme.Torso),              // shoulders, hips
    Triple(5, 11, Theme.Torso), Triple(6, 12, Theme.Torso),              // trunk
    Triple(0, 5, Theme.Torso.copy(alpha = 0.6f)),                        // neck
    Triple(0, 6, Theme.Torso.copy(alpha = 0.6f)),
)

/** 2D COCO-17 skeleton, color-coded left/right/torso. */
fun DrawScope.drawSkeleton(keypoints: List<Keypoint2D>, fit: Rect) {
    if (keypoints.size != 17) return
    val threshold = AppConfig.KPT_CONF_THRESHOLD
    fun p(i: Int) = VideoFitMapper.point(keypoints[i].x, keypoints[i].y, fit)

    for ((a, b, color) in bones) {
        if (keypoints[a].conf <= threshold || keypoints[b].conf <= threshold) continue
        drawLine(color, p(a), p(b), strokeWidth = 3f, cap = StrokeCap.Round)
    }
    for (i in keypoints.indices) {
        if (keypoints[i].conf <= threshold) continue
        val color = if (i == 0) Theme.Torso else if (i % 2 == 1) Theme.LeftSide else Theme.RightSide
        drawCircle(color, radius = if (i == 0) 5f else 4f, center = p(i))
    }
}
