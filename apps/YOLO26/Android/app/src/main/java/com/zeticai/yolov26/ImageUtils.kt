package com.zeticai.yolov26

import android.graphics.Bitmap
import android.graphics.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImageUtils {
    fun resize(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    /**
     * Prepares input for YOLO model:
     * 1. Resizes to target size (640x640)
     * 2. Normalizes pixel values (0-255 -> 0.0-1.0)
     * 3. Converts to CHW format Float Buffer
     */
    /**
     * Prepares input for YOLO model:
     * 1. Resizes to target size (640x640)
     * 2. Normalizes pixel values (0-255 -> 0.0-1.0)
     * 3. write to FloatBuffer directly
     */
    fun prepareInput(bitmap: Bitmap, targetWidth: Int = 640, targetHeight: Int = 640): FloatArray {
        // Optimized to minimize allocation where possible, 
        // though FloatArray return forces one allocation. 
        // For max performance, caller should pass a buffer.
        
        val resized = if (bitmap.width != targetWidth || bitmap.height != targetHeight) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }
        
        val pixels = IntArray(targetWidth * targetHeight)
        resized.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        val chwArray = FloatArray(3 * targetWidth * targetHeight)
        val area = targetWidth * targetHeight

        // Use Loop Unrolling or simple parallel if needed? 
        // Standard loop is usually fine if allocations are low.
        // But main gain is avoiding Bitmap.createScaledBitmap if size matches.
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            chwArray[i] = r
            chwArray[area + i] = g
            chwArray[area * 2 + i] = b
        }
        
        // Recycle if we created a new bitmap
        if (resized != bitmap) {
            resized.recycle()
        }
        
        return chwArray
    }
    
    // Optimized overload taking a reusable buffer
    fun prepareInputToBuffer(bitmap: Bitmap, outputBuffer: FloatArray, targetWidth: Int = 640, targetHeight: Int = 640) {
         val resized = if (bitmap.width != targetWidth || bitmap.height != targetHeight) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }
        
        val pixels = IntArray(targetWidth * targetHeight) // Could be cached too but it's fast
        resized.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)
        
        val area = targetWidth * targetHeight
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            outputBuffer[i] = r
            outputBuffer[area + i] = g
            outputBuffer[area * 2 + i] = b
        }
        
        if (resized != bitmap) {
            resized.recycle()
        }
    }
}
