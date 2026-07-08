package ai.zetic.demo.posemotion.pipeline

import ai.zetic.demo.posemotion.AppConfig
import android.graphics.RectF

/**
 * Decodes YOLO26n output. Layout auto-detected by element count (Tensor shape is
 * private in the Android SDK): 1800 floats = NMS-free export [1,300,6] with rows
 * (x1,y1,x2,y2,score,class) in 0..640 pixels; otherwise the raw [1,4+C,anchors] head.
 */
object DetectorDecoder {

    fun decode(data: FloatArray): Pair<Detection?, Detection?> {
        val detections = if (data.size % 6 == 0 && data.size <= 300 * 6) {
            decodeNms(data)
        } else {
            // Raw [1, 4+C, anchors]: for COCO C=80 → 84 rows; anchors = size / 84.
            val anchors = data.size / 84
            decodeRaw(data, classCount = 80, anchors = anchors)
        }

        var person: Detection? = null
        var ball: Detection? = null
        for (d in detections) {
            if (d.classIndex == AppConfig.PERSON_CLASS_INDEX &&
                d.score > (person?.score ?: 0f)) person = d
            if (d.classIndex == AppConfig.BALL_CLASS_INDEX &&
                d.score > (ball?.score ?: 0f)) ball = d
        }
        return person to ball
    }

    private fun decodeNms(data: FloatArray): List<Detection> {
        val size = AppConfig.DET_SIZE.toFloat()
        val rows = data.size / 6
        val result = ArrayList<Detection>()
        for (i in 0 until rows) {
            val o = i * 6
            val score = data[o + 4]
            if (score <= AppConfig.DET_CONF_THRESHOLD) continue
            val rect = RectF(
                data[o] / size,
                data[o + 1] / size,
                data[o + 2] / size,
                data[o + 3] / size,
            )
            result.add(Detection(rect, score, data[o + 5].toInt()))
        }
        return result
    }

    private fun decodeRaw(data: FloatArray, classCount: Int, anchors: Int): List<Detection> {
        val size = AppConfig.DET_SIZE.toFloat()
        // Only the two classes the demo uses — keeps the scan cheap.
        val wanted = intArrayOf(AppConfig.PERSON_CLASS_INDEX, AppConfig.BALL_CLASS_INDEX)
        val result = ArrayList<Detection>()
        for (a in 0 until anchors) {
            for (c in wanted) {
                if (c >= classCount) continue
                val score = data[(4 + c) * anchors + a]
                if (score <= AppConfig.DET_CONF_THRESHOLD) continue
                val xc = data[a]
                val yc = data[anchors + a]
                val w = data[2 * anchors + a]
                val h = data[3 * anchors + a]
                val rect = RectF(
                    (xc - w / 2) / size,
                    (yc - h / 2) / size,
                    (xc + w / 2) / size,
                    (yc + h / 2) / size,
                )
                result.add(Detection(rect, score, c))
            }
        }
        return nms(result)
    }

    private fun nms(boxes: List<Detection>): List<Detection> {
        val sorted = boxes.sortedByDescending { it.score }
        val kept = ArrayList<Detection>()
        for (box in sorted) {
            val overlaps = kept.any {
                it.classIndex == box.classIndex && iou(it.rect, box.rect) > AppConfig.IOU_THRESHOLD
            }
            if (!overlaps) kept.add(box)
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix = maxOf(0f, minOf(a.right, b.right) - maxOf(a.left, b.left))
        val iy = maxOf(0f, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top))
        val inter = ix * iy
        if (inter <= 0f) return 0f
        return inter / (a.width() * a.height() + b.width() * b.height() - inter)
    }
}
