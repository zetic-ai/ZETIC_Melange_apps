package ai.zetic.demo.posemotion.video

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat

data class DecodedFrame(val bitmap: Bitmap, val ptsUs: Long)

/**
 * Deterministic frame-by-frame H.264 decode: MediaExtractor + MediaCodec in
 * synchronous ByteBuffer mode. `nextFrame()` returns exactly one frame per call
 * (pull-based — the codec's small buffer pool provides natural backpressure),
 * or null at end of stream (caller rewinds to loop). Confined to one thread.
 */
class VideoFrameDecoder(private val source: ClipSource) {
    private var extractor: MediaExtractor? = null
    private var codec: MediaCodec? = null
    private var inputDone = false
    private val yuvToRgb = YuvToRgb()

    // Triple-buffered output bitmaps: by the time one is reused, two newer frames
    // have been posted, so Compose is no longer drawing it.
    private val pool = arrayOfNulls<Bitmap>(3)
    private var poolIndex = 0

    fun open() {
        val ex = MediaExtractor()
        when (source) {
            is ClipSource.Asset ->
                ex.setDataSource(source.afd.fileDescriptor, source.afd.startOffset, source.afd.length)
            is ClipSource.Local -> ex.setDataSource(source.path)
        }
        var track = -1
        var format: MediaFormat? = null
        for (i in 0 until ex.trackCount) {
            val f = ex.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                track = i
                format = f
                break
            }
        }
        val fmt = format ?: throw IllegalStateException("no video track in clip")
        ex.selectTrack(track)
        fmt.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
        )
        val mime = fmt.getString(MediaFormat.KEY_MIME)!!
        val c = MediaCodec.createDecoderByType(mime)
        c.configure(fmt, null, null, 0)
        c.start()
        extractor = ex
        codec = c
        inputDone = false
    }

    /** Decodes and returns the next frame, or null at EOS. */
    fun nextFrame(): DecodedFrame? {
        val codec = codec ?: return null
        val extractor = extractor ?: return null
        val info = MediaCodec.BufferInfo()

        var spins = 0
        while (spins < 500) {   // ~5 s worst case; a healthy decode exits in a few loops
            if (!inputDone) {
                val inIdx = codec.dequeueInputBuffer(10_000)
                if (inIdx >= 0) {
                    val buf = codec.getInputBuffer(inIdx)!!
                    val size = extractor.readSampleData(buf, 0)
                    if (size < 0) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            when (val outIdx = codec.dequeueOutputBuffer(info, 10_000)) {
                MediaCodec.INFO_TRY_AGAIN_LATER,
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> spins++
                else -> if (outIdx >= 0) {
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        codec.releaseOutputBuffer(outIdx, false)
                        return null
                    }
                    if (info.size == 0) {
                        codec.releaseOutputBuffer(outIdx, false)
                        spins++
                        continue
                    }
                    val image = codec.getOutputImage(outIdx)
                    if (image == null) {
                        codec.releaseOutputBuffer(outIdx, false)
                        spins++
                        continue
                    }
                    val bitmap = acquireBitmap(image.cropRect.width(), image.cropRect.height())
                    yuvToRgb.convert(image, bitmap)
                    codec.releaseOutputBuffer(outIdx, false)   // returns buffer to the codec
                    return DecodedFrame(bitmap, info.presentationTimeUs)
                } else spins++
            }
        }
        return null   // treated as EOS; caller rewinds
    }

    /** Loop the clip: seek to start and flush the codec (legal in sync mode). */
    fun rewind() {
        extractor?.seekTo(0, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
        codec?.flush()
        inputDone = false
    }

    fun release() {
        runCatching { codec?.stop() }
        runCatching { codec?.release() }
        runCatching { extractor?.release() }
        (source as? ClipSource.Asset)?.let { runCatching { it.afd.close() } }
        codec = null
        extractor = null
    }

    private fun acquireBitmap(w: Int, h: Int): Bitmap {
        val existing = pool[poolIndex]
        val bitmap = if (existing != null && existing.width == w && existing.height == h) {
            existing
        } else {
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { pool[poolIndex] = it }
        }
        poolIndex = (poolIndex + 1) % pool.size
        return bitmap
    }
}
