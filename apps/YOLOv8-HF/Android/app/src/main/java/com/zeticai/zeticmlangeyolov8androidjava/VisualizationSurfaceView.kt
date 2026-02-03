package com.zeticai.zeticmlangeyolov8androidjava

import android.content.Context
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import com.zeticai.mlange.feature.yolov8.YoloResult
import java.io.BufferedReader
import java.io.InputStreamReader
import android.util.Size

class VisualizationSurfaceView(
    context: Context,
    attrSet: AttributeSet,
) : PreviewSurfaceView(context, attrSet) {
    private val yoloClasses: List<String> by lazy {
        context.assets.open("coco.yaml").use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                reader.readLines().drop(17).map {
                    it//.split(": ")[1]
                }
            }
        }
    }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.ADD)
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.GREEN
        textSize = 40f
    }

    init {
        setBackgroundColor(Color.TRANSPARENT)
        setZOrderOnTop(true)
        holder.setFormat(PixelFormat.TRANSPARENT)
    }

    fun visualize(
        yoloResult: YoloResult?, inputSize: Size, isRotated: Boolean
    ) {
        if (!holder.surface.isValid) return

        if (yoloResult == null) return

        if (yoloResult.value.isEmpty()) return

        if (yoloClasses.isEmpty()) return

        val canvas = holder.lockCanvas()
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        val inputRect = if (!isRotated) {
            Rect(0, 0, inputSize.width, inputSize.height)
        } else {
            Rect(0, 0, inputSize.height, inputSize.width)
        }
        val targetRect = Rect(
            0, 0, holder.surfaceFrame.width(), holder.surfaceFrame.height()
        )

        yoloResult.value.forEach {
            val convertedMin = transformCoordToTargetCoord(Pair(it.box.xMin, it.box.yMin), inputRect, targetRect)
            val convertedMax = transformCoordToTargetCoord(Pair(it.box.xMax, it.box.yMax), inputRect, targetRect)

            val rect = RectF(
                convertedMin.first,
                convertedMin.second,
                convertedMax.first,
                convertedMax.second,
            )
            canvas.drawRect(rect, paint)
            canvas.drawText(
                "${yoloClasses[it.classId]} : ${it.confidence}",
                rect.left,
                rect.top - 10f,
                paint
            )
        }
        holder.unlockCanvasAndPost(canvas)
    }
}