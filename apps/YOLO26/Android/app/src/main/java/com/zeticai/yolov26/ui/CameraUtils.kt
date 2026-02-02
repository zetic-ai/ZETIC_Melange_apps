package com.zeticai.yolov26.ui

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.util.Range
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider

object CameraUtils {
    
    data class CameraSelection(
        val cameraInfo: CameraInfo?,
        val fpsRange: Range<Int>?
    )

    fun findBestHighFPSCamera(cameraProvider: ProcessCameraProvider, lensFacing: Int = CameraSelector.LENS_FACING_BACK): CameraSelection {
        var bestCameraInfo: CameraInfo? = null
        var bestFpsRange: Range<Int>? = null
        var maxFps = 0
        
        // Re-using existing logic to return the "Best" initial selection
        var selection = CameraSelection(null, null)
        
        for (cameraInfo in cameraProvider.availableCameraInfos) {
             try {
                if (isLensFacing(cameraInfo, lensFacing)) {
                    val ranges = getRanges(cameraInfo)
                    ranges.forEach { range ->
                        if (range.upper > maxFps) {
                             maxFps = range.upper
                             bestFpsRange = range
                             bestCameraInfo = cameraInfo
                        }
                    }
                }
             } catch (e: Exception) {}
        }
        return CameraSelection(bestCameraInfo, bestFpsRange)
    }
    
    fun getAvailableFPSRanges(cameraProvider: ProcessCameraProvider, cameraInfo: CameraInfo?): List<Range<Int>> {
        val targetInfo = cameraInfo ?: return emptyList()
        return try {
            getRanges(targetInfo).sortedByDescending { it.upper }
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    private fun isLensFacing(cameraInfo: CameraInfo, lensFacing: Int): Boolean {
         return try {
             val settings = Camera2CameraInfo.from(cameraInfo)
             settings.getCameraCharacteristic(CameraCharacteristics.LENS_FACING) == 
                 if (lensFacing == CameraSelector.LENS_FACING_BACK) CameraCharacteristics.LENS_FACING_BACK 
                 else CameraCharacteristics.LENS_FACING_FRONT
         } catch(e: Exception) { false }
    }
    
    private fun isBackCamera(cameraInfo: CameraInfo): Boolean {
         return try {
             val settings = Camera2CameraInfo.from(cameraInfo)
             settings.getCameraCharacteristic(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
         } catch(e: Exception) { false }
    }
    
    private fun getRanges(cameraInfo: CameraInfo): Array<Range<Int>> {
        val settings = Camera2CameraInfo.from(cameraInfo)
        return settings.getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: emptyArray()
    }
    
    fun applyHighFPS(
        selection: CameraSelection,
        previewBuilder: Preview.Builder,
        imageAnalysisBuilder: ImageAnalysis.Builder
    ): String {
        val range = selection.fpsRange
        if (range != null && range.upper > 30) {
            Log.d("HighFPS", "Selected Camera FPS: $range")
            
            val extPreview = Camera2Interop.Extender(previewBuilder)
            extPreview.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, range)
            
            val extAnalysis = Camera2Interop.Extender(imageAnalysisBuilder)
            extAnalysis.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, range)
            
            return "High FPS: [${range.lower}, ${range.upper}]"
        }
        return "Standard FPS (30)"
    }
}
