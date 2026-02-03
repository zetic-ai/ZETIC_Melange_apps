package com.zeticai.zeticmlangeyolov8androidjava

import android.content.Context
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Size
import android.view.Gravity
import android.view.SurfaceView
import android.widget.FrameLayout
import kotlin.math.min

open class PreviewSurfaceView(context: Context, attrSet: AttributeSet) :
    SurfaceView(context, attrSet) {

    fun updateSizeKeepRatio(size: Size) {
        val metrics = resources.displayMetrics
        val screenSize = Size(metrics.widthPixels, metrics.heightPixels)

        val rotatedSize = Size(size.height, size.width)
        holder.setFixedSize(rotatedSize.width, rotatedSize.height)

        val imageRatio = rotatedSize.width.toFloat() / rotatedSize.height.toFloat()
        val screenRatio = screenSize.width.toFloat() / screenSize.height.toFloat()

        val newSize = if (imageRatio > screenRatio) {
            Size(
                screenSize.width,
                (screenSize.width / imageRatio).toInt()
            )
        } else {
            Size(
                (screenSize.height * imageRatio).toInt(),
                screenSize.height
            )
        }

        updateSize(newSize)
    }

    private fun updateSize(size: Size) {
        layoutParams = FrameLayout.LayoutParams(size.width, size.height, Gravity.CENTER)
    }

    protected fun transformCoordToTargetCoord(
        coord: Pair<Float, Float>,
        originalSize: Rect,
        targetSize: Rect
    ): Pair<Float, Float> {
        val originalWidth = originalSize.width()
        val originalHeight = originalSize.height()
        val originalCenterX = originalWidth / 2
        val originalCenterY = originalHeight / 2

        val targetWidth = targetSize.width()
        val targetHeight = targetSize.height()
        val targetCenterX = targetWidth / 2
        val targetCenterY = targetHeight / 2

        val widthRatio = targetWidth.toFloat() / originalWidth.toFloat()
        val heightRatio = targetHeight.toFloat() / originalHeight.toFloat()

        val resizeFactor = min(widthRatio, heightRatio)
        val retX = ((coord.first - originalCenterX) * resizeFactor + targetCenterX)
        val retY = ((coord.second - originalCenterY) * resizeFactor + targetCenterY)

        return Pair(retX, retY)
    }

    protected fun transformRectToTargetRect(
        rect: RectF,
        originalRect: Rect,
        targetRect: Rect
    ): RectF {
        val convertedMin =
            transformCoordToTargetCoord(rect.left to rect.top, originalRect, targetRect)
        val convertedMax =
            transformCoordToTargetCoord(rect.right to rect.bottom, originalRect, targetRect)

        return RectF(
            convertedMin.first,
            convertedMin.second,
            convertedMax.first,
            convertedMax.second,
        )
    }
}