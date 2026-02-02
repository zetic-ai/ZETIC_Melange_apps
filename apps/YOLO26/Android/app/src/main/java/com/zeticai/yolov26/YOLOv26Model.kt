package com.zeticai.yolov26

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.tensor.Tensor
import com.zeticai.mlange.core.tensor.DataType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YOLOv26Model(context: Context) {
    private var model: ZeticMLangeModel? = null
    
    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded = _isModelLoaded.asStateFlow()
    
    // Output state
    private val _detectionResults = MutableStateFlow<List<BoundingBox>>(emptyList())
    val detectionResults = _detectionResults.asStateFlow()
    
    private val _inferenceTime = MutableStateFlow(0L)
    val inferenceTime = _inferenceTime.asStateFlow()
    
    // UI needs source size to calculate aspect ratio scaling (Fit vs Fill)
    private val _sourceImageSize = MutableStateFlow(Pair(1, 1))
    val sourceImageSize = _sourceImageSize.asStateFlow()

    init {
        // Load Model Asynchronously
        // "dev_d786c1fd7f2848acb9b0bf8060aa10b2", "Team_ZETIC/YOLOv26"
        loadModel(context)
    }

    private fun loadModel(context: Context) {
        // Run on background thread
        // Note: In real app use Coroutines Scope properly. Here init simple.
        Thread {
            try {
                // Using version 3 as per iOS config
                model = ZeticMLangeModel(context, "YOUR_MLANGE_KEY", "Team_ZETIC/YOLOv26", 3)
                // Simulate loading
                // Thread.sleep(1000)
                _isModelLoaded.value = true
                Log.d("YOLOv26", "Model Loaded Successfully")
            } catch (e: Exception) {
                Log.e("YOLOv26", "Failed to load model", e)
            }
        }.start()
    }

    // Reusable buffers for Zero-Allocation pipeline
    private val inputBuffer = FloatArray(3 * 640 * 640)
    // ByteBuffer for model input (size * 4 bytes for Float)
    private val inputByteBuffer: ByteBuffer = ByteBuffer.allocateDirect(3 * 640 * 640 * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    
    private val pixelBuffer = IntArray(640 * 640)
    private val resizedBitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
    private val canvas = android.graphics.Canvas(resizedBitmap)
    private val matrix = android.graphics.Matrix()
    private val paint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)

    // Cached View to avoid allocation per frame
    private val inputFloatBuffer = inputByteBuffer.asFloatBuffer()

    suspend fun detect(bitmap: Bitmap) = withContext(Dispatchers.Default) {
        if (!_isModelLoaded.value) return@withContext

        val startTime = System.currentTimeMillis()
        
        // ... (lines 74-90 same as original)
        // Update source size for UI
        _sourceImageSize.value = Pair(bitmap.width, bitmap.height)
        
        // 1. Resize & Draw directly to pre-allocated bitmap (Minimizing Copy)
        // Calculate scale manually to fit 640x640
        matrix.reset()
        val scaleX = 640f / bitmap.width
        val scaleY = 640f / bitmap.height
        matrix.setScale(scaleX, scaleY)
        
        // "Merge" resize operation into a single draw call on reused storage
        canvas.drawColor(android.graphics.Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR) // generic clear if needed, though we overwrite
        canvas.drawBitmap(bitmap, matrix, paint)
        
        // 2. Extract Pixels to pre-allocated buffer
        resizedBitmap.getPixels(pixelBuffer, 0, 640, 0, 0, 640, 640)
        
        // 3. Convert to CHW Float Planar
        // Optimization: Use multiplication and pre-calculated offsets
        val area = 640 * 640
        val offsetG = area
        val offsetB = area * 2
        val norm = 1.0f / 255.0f
        
        for (i in 0 until area) {
            val pixel = pixelBuffer[i]
            
            // Optimized: bit extraction + multiplication
            inputBuffer[i]           = ((pixel shr 16) and 0xFF) * norm // R
            inputBuffer[offsetG + i] = ((pixel shr 8) and 0xFF) * norm  // G
            inputBuffer[offsetB + i] = (pixel and 0xFF) * norm          // B
        }
        
        // 2. Wrap in Tensor using ByteBuffer
        inputFloatBuffer.clear()
        inputFloatBuffer.put(inputBuffer)
        
        val inputTensor = Tensor(
            inputByteBuffer,
            DataType.Float32,
            intArrayOf(1, 3, 640, 640)
        )
        val inputs = arrayOf(inputTensor)
        
        try {
            // 3. Run Inference
            val outputs = model?.run(inputs)
            
            val outputTensor = outputs?.firstOrNull() ?: return@withContext
            
            // 4. Post Process
            // data property returns ByteBuffer
            val outputBuffer = outputTensor.data
            outputBuffer.rewind()
            
            // Infer shape from size since 'shape' property is private
            // Float32 = 4 bytes
            val floats = outputBuffer.remaining() / 4
            val outputData = FloatArray(floats)
            outputBuffer.asFloatBuffer().get(outputData)
            
            val rows: Int
            val cols: Int
            
            // Heuristic shape inference
            if (floats == 1800) { 
                // [1, 300, 6] -> NMS output
                rows = 300
                cols = 6
            } else if (floats == 705600) {
                // [1, 84, 8400] -> Raw output
                 rows = 84
                 cols = 8400
            } else {
                 // Fallback or attempt to guess NMS output size
                 // Just try treating as NMS if small
                 rows = floats / 6
                 cols = 6
            }
            
            val results = PostProcess.process(
                outputData, 
                rows, 
                cols, 
                bitmap.width.toFloat(), 
                bitmap.height.toFloat()
            )
            
            val duration = System.currentTimeMillis() - startTime
            
            _detectionResults.emit(results)
            _inferenceTime.emit(duration)
            
        } catch (e: Exception) {
            Log.e("YOLOv26", "Inference error", e)
            _detectionResults.emit(emptyList())
        }
    }
}
