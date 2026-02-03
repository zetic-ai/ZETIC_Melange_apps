package com.zeticai.zeticmlangeyolov8androidjava.feature

import android.content.Context
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.model.ZeticMLangeTarget
import com.zeticai.mlange.feature.yolov8.YoloResult
import com.zeticai.mlange.feature.yolov8.Yolov8Wrapper
import java.io.File
import java.io.FileOutputStream

class YOLOv8(
    context: Context,
) {
    private val model: ZeticMLangeModel = ZeticMLangeModel(
        context,
        "YOUR_MLANGE_KEY",
        MODEL_KEY
    )
    private val wrapper: Yolov8Wrapper

    init {
        val cocoYamlSamplePath = "coco.yaml"
        copyFileFromAssetsToData(context, cocoYamlSamplePath)
        val cocoYamlFile = File(context.filesDir, cocoYamlSamplePath)
        val cocoYamlPath = cocoYamlFile.absolutePath
        wrapper = Yolov8Wrapper(cocoYamlPath)
    }

    fun run(imagePtr: Long): YoloResult {
        val preprocess = wrapper.preprocess(imagePtr)
        model.run(arrayOf(preprocess))
        return wrapper.postprocess(model.outputBuffers[0].array())
    }

    fun close() {
        model.deinit()
        wrapper.deinit()
    }

    private fun copyFileFromAssetsToData(context: Context, fileName: String) {
        val assetManager = context.assets
        val file = assetManager.open(fileName)
        val outFile = File(context.filesDir, fileName)
        val out = FileOutputStream(outFile)

        val buffer = ByteArray(1024)
        var read: Int
        while ((file.read(buffer).also { read = it }) != -1) {
            out.write(buffer, 0, read)
        }

        file.close()
        out.flush()
        out.close()
    }

    companion object {
        const val MODEL_KEY: String = "b9f5d74e6f644288a32c50174ded828e"
    }
}