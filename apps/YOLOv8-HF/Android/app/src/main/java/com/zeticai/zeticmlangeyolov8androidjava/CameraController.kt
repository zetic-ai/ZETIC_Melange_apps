package com.zeticai.zeticmlangeyolov8androidjava

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.Surface
import android.view.SurfaceHolder
import androidx.core.app.ActivityCompat

class CameraController @JvmOverloads constructor(
    context: Context,
    private val previewSurfaceView: PreviewSurfaceView,
    private val visualizationSurfaceView: PreviewSurfaceView,
    private val onCameraFrame: (ByteArray, Size) -> Unit,
    private val onSurface: (Surface) -> Unit,
    cameraDirection: CameraDirection = CameraDirection.BACK
) {
    private val manager: CameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val cameraId: String = manager.cameraIdList[cameraDirection.id]
    private val handler = Handler(
        HandlerThread("camera2").apply {
            start()
        }.looper
    )

    private val yoloInputSize: Size = getSizeForMinResolution(context, 640)
    private val imageReader = ImageReader.newInstance(
        yoloInputSize.width, yoloInputSize.height, ImageFormat.JPEG, 2
    )

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private val cameraDeviceStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            cameraDevice?.createCaptureSession(
                listOf(
                    previewSurfaceView.holder.surface,
                    imageReader.surface,
                ), cameraCaptureSessionStateCallback, handler
            )
            previewSurfaceView.invalidate()
        }

        override fun onDisconnected(camera: CameraDevice) {
            cameraDevice?.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {

        }
    }

    private val previewBuilder by lazy {
        cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(imageReader.surface)
            addTarget(previewSurfaceView.holder.surface)
        }
    }

    private val cameraCaptureSessionStateCallback = object : CameraCaptureSession.StateCallback() {
        override fun onConfigured(session: CameraCaptureSession) {
            captureSession = session
            session.setRepeatingRequest(previewBuilder.build(), null, handler)
        }

        override fun onConfigureFailed(session: CameraCaptureSession) {

        }
    }
    private val previewSurfaceHolderCallback = object : SurfaceHolder.Callback {
        override fun surfaceCreated(holder: SurfaceHolder) {
        }

        override fun surfaceChanged(
            holder: SurfaceHolder, format: Int, width: Int, height: Int
        ) {
            onSurface(holder.surface)
            if (ActivityCompat.checkSelfPermission(
                    context, Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) throw IllegalArgumentException()
            manager.openCamera(cameraId, cameraDeviceStateCallback, handler)
        }

        override fun surfaceDestroyed(holder: SurfaceHolder) {
            cameraDevice?.close()
        }
    }

    fun startPreview() {
        visualizationSurfaceView.updateSizeKeepRatio(yoloInputSize)
        previewSurfaceView.updateSizeKeepRatio(yoloInputSize)
        imageReader.setOnImageAvailableListener(
            {
                val image = it.acquireLatestImage() ?: return@setOnImageAvailableListener
                processCameraImage(image)
                image.close()
            }, handler
        )

        previewSurfaceView.holder.addCallback(previewSurfaceHolderCallback)
    }

    private fun processCameraImage(image: Image) {
        val buffer = image.planes[0].buffer
        val array = ByteArray(buffer.remaining())
        buffer.get(array)

        onCameraFrame(array, yoloInputSize)
    }

    private fun getSizeForMinResolution(context: Context , minDimension: Int): Size {
        val manager: CameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId: String = manager.cameraIdList[0]
        val characteristics: CameraCharacteristics = manager.getCameraCharacteristics(cameraId)

        val sizes = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?.getOutputSizes(ImageFormat.JPEG)

        if (sizes.isNullOrEmpty()) {
            throw Exception("No camera found")
        }

        for (i in sizes.size - 1 downTo 0) {
            val size = sizes[i]
            if (size.width >= minDimension && size.height >= minDimension) {
                return size
            }
        }

        throw Exception("No size found")
    }

    fun close() {
        cameraDevice?.close()
        captureSession?.close()
        imageReader.close()
    }

    companion object {
        const val ROTATE_COUNTER_CLOCKWISE = -90
        const val ROTATE_CLOCKWISE = 90
    }
}