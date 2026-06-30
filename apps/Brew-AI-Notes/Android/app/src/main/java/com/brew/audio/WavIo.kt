package com.brew.audio

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * PCM16 mono WAV reading/writing at a fixed sample rate. Mirrors VoxScribe's
 * `preprocessor.dart` decode (int16 / float32, ch0 downmix, linear resample to
 * 16 kHz) so the bytes that reach Whisper are bit-compatible with that pipeline.
 */
object WavIo {

    const val TARGET_SAMPLE_RATE = 16_000

    /** Streaming WAV writer: 44-byte header up front, sizes back-patched on close. */
    class Writer(file: File, private val sampleRate: Int = TARGET_SAMPLE_RATE) : AutoCloseable {
        private val raf = RandomAccessFile(file, "rw")
        private var dataBytes = 0
        private val scratch = ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN)

        init {
            raf.setLength(0)
            raf.write(ByteArray(44)) // placeholder header
        }

        /** Append float PCM samples in [-1,1], converting to int16 LE. */
        fun write(samples: FloatArray, count: Int) {
            var i = 0
            while (i < count) {
                scratch.clear()
                while (i < count && scratch.remaining() >= 2) {
                    val v = (samples[i].coerceIn(-1f, 1f) * 32767f).toInt()
                    scratch.putShort(v.toShort())
                    i++
                }
                raf.write(scratch.array(), 0, scratch.position())
                dataBytes += scratch.position()
            }
        }

        override fun close() {
            patchHeader()
            raf.close()
        }

        private fun patchHeader() {
            val byteRate = sampleRate * 2 // mono, 16-bit
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray(Charsets.US_ASCII))
            header.putInt(36 + dataBytes)
            header.put("WAVE".toByteArray(Charsets.US_ASCII))
            header.put("fmt ".toByteArray(Charsets.US_ASCII))
            header.putInt(16)             // PCM fmt chunk size
            header.putShort(1)            // audioFormat = PCM
            header.putShort(1)            // channels = mono
            header.putInt(sampleRate)
            header.putInt(byteRate)
            header.putShort(2)            // block align
            header.putShort(16)           // bits per sample
            header.put("data".toByteArray(Charsets.US_ASCII))
            header.putInt(dataBytes)
            raf.seek(0)
            raf.write(header.array())
        }
    }

    /** Decode a WAV file to mono float [-1,1] resampled to 16 kHz. */
    fun readMono16k(file: File): FloatArray {
        val bytes = file.readBytes()
        val (samples, rate, channels) = decodeWav(bytes)
        val mono = downmixCh0(samples, channels)
        return resampleLinear(mono, rate, TARGET_SAMPLE_RATE)
    }

    private data class Decoded(val samples: FloatArray, val sampleRate: Int, val channels: Int)

    private fun tag(b: ByteArray, o: Int) = String(b, o, 4, Charsets.US_ASCII)

    private fun decodeWav(bytes: ByteArray): Decoded {
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size >= 12 && tag(bytes, 0) == "RIFF" && tag(bytes, 8) == "WAVE") {
            "Not a RIFF/WAVE file"
        }
        var offset = 12
        var audioFormat = 1
        var channels = 1
        var sampleRate = TARGET_SAMPLE_RATE
        var bits = 16
        var dataStart = -1
        var dataLen = 0
        while (offset + 8 <= bytes.size) {
            val id = tag(bytes, offset)
            val size = bb.getInt(offset + 4)
            val body = offset + 8
            when (id) {
                "fmt " -> {
                    audioFormat = bb.getShort(body).toInt() and 0xFFFF
                    channels = bb.getShort(body + 2).toInt() and 0xFFFF
                    sampleRate = bb.getInt(body + 4)
                    bits = bb.getShort(body + 14).toInt() and 0xFFFF
                }
                "data" -> {
                    dataStart = body
                    dataLen = size
                }
            }
            offset = body + size + (size and 1) // word-aligned chunks
        }
        require(dataStart >= 0) { "No data chunk" }
        if (dataStart + dataLen > bytes.size) dataLen = bytes.size - dataStart

        val out: FloatArray
        when {
            audioFormat == 3 && bits == 32 -> {
                val n = dataLen / 4
                out = FloatArray(n) { bb.getFloat(dataStart + it * 4) }
            }
            audioFormat == 1 && bits == 16 -> {
                val n = dataLen / 2
                out = FloatArray(n) { bb.getShort(dataStart + it * 2) / 32768f }
            }
            else -> throw IllegalArgumentException("Unsupported WAV: fmt=$audioFormat bits=$bits")
        }
        return Decoded(out, sampleRate, channels)
    }

    private fun downmixCh0(interleaved: FloatArray, channels: Int): FloatArray {
        if (channels <= 1) return interleaved
        val frames = interleaved.size / channels
        return FloatArray(frames) { interleaved[it * channels] }
    }

    private fun resampleLinear(input: FloatArray, inRate: Int, outRate: Int): FloatArray {
        if (inRate == outRate || input.isEmpty()) return input.copyOf()
        val outLen = Math.round(input.size.toLong() * outRate / inRate.toDouble()).toInt()
        if (outLen <= 1) return floatArrayOf(input.first())
        val out = FloatArray(outLen)
        val step = (input.size - 1).toDouble() / (outLen - 1)
        for (i in 0 until outLen) {
            val pos = i * step
            val i0 = pos.toInt()
            val i1 = if (i0 + 1 < input.size) i0 + 1 else i0
            val frac = (pos - i0).toFloat()
            out[i] = input[i0] * (1 - frac) + input[i1] * frac
        }
        out[0] = input.first()
        out[outLen - 1] = input.last()
        return out
    }
}
