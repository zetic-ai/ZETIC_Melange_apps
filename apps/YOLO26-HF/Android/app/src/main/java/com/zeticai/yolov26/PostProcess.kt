package com.zeticai.yolov26

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val score: Float,
    val classIndex: Int,
    val label: String
)

object PostProcess {
    private const val CONFIDENCE_THRESHOLD = 0.25f
    private const val IOU_THRESHOLD = 0.45f
    
    // COCO Classes
    val CLASSES = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    )

    fun process(
        outputData: FloatArray, 
        rows: Int, 
        cols: Int, 
        imgWidth: Float, 
        imgHeight: Float
    ): List<BoundingBox> {
        val boxes = ArrayList<BoundingBox>()
        
        // Check input shape to decide safe parsing strategy
        // Standard NMS output from CoreML export was [1, 300, 6] -> Rows=300, Cols=6
        // Raw output is [1, 84, 8400] -> Rows=84, Cols=8400 (Need Transpose logic if raw)
        
        // Implementing Standard NMS Output logic first (matching iOS PostProcessNMS)
        // [Batch, N, 6] -> [x1, y1, x2, y2, score, class]
        
        if (cols == 6) {
           for (i in 0 until rows) {
               val offset = i * cols
               if (offset + 5 >= outputData.size) break
               
               val score = outputData[offset + 4]
               if (score > CONFIDENCE_THRESHOLD) {
                   val classIdx = outputData[offset + 5].toInt()
                   
                   val x1 = outputData[offset + 0]
                   val y1 = outputData[offset + 1]
                   val x2 = outputData[offset + 2]
                   val y2 = outputData[offset + 3]
                   
                   // Normalize to 0-1 (Assuming model output is 0-640 absolute)
                   val nx1 = x1 / 640f
                   val ny1 = y1 / 640f
                   val nx2 = x2 / 640f
                   val ny2 = y2 / 640f
                   
                   // Calculate dimensions in normalized space
                   val w = nx2 - nx1
                   val h = ny2 - ny1
                   val cx = nx1 + w / 2
                   val cy = ny1 + h / 2
                   
                   val label = if (classIdx in CLASSES.indices) CLASSES[classIdx] else "Class $classIdx"
                   
                   boxes.add(BoundingBox(
                       nx1, ny1, nx2, ny2,
                       cx, cy, w, h,
                       score, classIdx, label
                   ))
               }
           }
        }
        
        // TODO: Raw Output Handling if needed (mirrors iOS postprocess)
        // For now start with NMS assumption as per "Correct Model" findings
        
        return boxes // NMS already applied by model if using exported model
    }
}
