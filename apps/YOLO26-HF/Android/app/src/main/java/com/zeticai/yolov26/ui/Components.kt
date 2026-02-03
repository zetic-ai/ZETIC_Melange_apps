package com.zeticai.yolov26.ui

import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.NativeCanvas
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import com.zeticai.yolov26.BoundingBox

@Composable
fun BoundingBoxOverlay(
    boxes: List<BoundingBox>,
    sourceWidth: Int,
    sourceHeight: Int,
    isFill: Boolean = false, // true = Crop/Zoom (Camera), false = Fit/Letterbox (Photo)
    alignmentTopStart: Boolean = false // Special case for Camera FILL_START
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val screenWidth = size.width
        val screenHeight = size.height
        
        // Calculate Scale Factor
        val scale: Float
        val offsetX: Float
        val offsetY: Float
        
        val srcRatio = sourceWidth.toFloat() / sourceHeight.toFloat()
        val screenRatio = screenWidth / screenHeight
        

        
        if (isFill) {
            // SCALE_TYPE_FILL (Crop)
            // Scale to cover the entire screen
            // If screen is wider than source, scale by width
            // If screen is taller than source, scale by height
            // Logic: max(scaleX, scaleY)
            val scaleX = screenWidth / sourceWidth
            val scaleY = screenHeight / sourceHeight
            scale = kotlin.math.max(scaleX, scaleY)
            
            val renderedWidth = sourceWidth * scale
            val renderedHeight = sourceHeight * scale
            
            if (alignmentTopStart) {
                // FILL_START
                offsetX = 0f
                offsetY = 0f
            } else {
                // FILL_CENTER (Standard)
                offsetX = (screenWidth - renderedWidth) / 2f
                offsetY = (screenHeight - renderedHeight) / 2f
            }
        } else {
            // SCALE_TYPE_FIT (Letterbox)
            // Scale to fit within the screen
            // min(scaleX, scaleY)
            val scaleX = screenWidth / sourceWidth
            val scaleY = screenHeight / sourceHeight
            scale = kotlin.math.min(scaleX, scaleY)
            
            val renderedWidth = sourceWidth * scale
            val renderedHeight = sourceHeight * scale
            
            // Usually FIT_CENTER
            offsetX = (screenWidth - renderedWidth) / 2f
            offsetY = (screenHeight - renderedHeight) / 2f
        }
        
        for (box in boxes) {
            drawBox(box, scale, offsetX, offsetY, sourceWidth, sourceHeight)
        }
    }
}

fun DrawScope.drawBox(
    box: BoundingBox, 
    scaleReference: Float, 
    offsetX: Float, 
    offsetY: Float,
    sourceW: Int,
    sourceH: Int
) {
    val colors = listOf(
        Color.Red, Color.Green, Color.Blue, Color.Cyan, Color.Magenta, Color.Yellow
    )
    val color = colors[box.classIndex % colors.size]

    // Normalize coordinates mapped to Source Image Size first
    // Then Scale and Offset to Screen
    val x1 = box.x1 * sourceW * scaleReference + offsetX
    val y1 = box.y1 * sourceH * scaleReference + offsetY
    val w = box.w * sourceW * scaleReference
    val h = box.h * sourceH * scaleReference
    
    /* 
       Optimization: simpler math using normalized coordinates directly
       x1_pixel = box.x1 * renderedWidth + offsetX
       renderedWidth = sourceW * scale
       -> x1_pixel = box.x1 * (sourceW * scale) + offsetX
       This matches logic above.
    */
    
    // Draw Rect
    drawRect(
        color = color,
        topLeft = Offset(x1, y1),
        size = Size(w, h),
        style = Stroke(width = 5f)
    )
    
    // Draw Text Background
    val paint = Paint().apply {
        this.color = android.graphics.Color.WHITE
        textSize = 40f
        typeface = Typeface.DEFAULT_BOLD
    }
    
    val text = "${box.label} ${String.format("%.2f", box.score)}"
    val textWidth = paint.measureText(text)
    val textHeight = 50f
    
    drawRect(
        color = color,
        topLeft = Offset(x1, y1 - textHeight),
        size = Size(textWidth + 20f, textHeight)
    )
    
    // Draw Text
    drawContext.canvas.nativeCanvas.drawText(
        text,
        x1 + 10f,
        y1 - 10f,
        paint
    )
}
