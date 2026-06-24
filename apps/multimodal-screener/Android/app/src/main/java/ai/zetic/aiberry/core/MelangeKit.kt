package ai.zetic.aiberry.core

import android.content.Context
import com.zeticai.mlange.core.cache.ModelCacheHandlingPolicy
import com.zeticai.mlange.core.model.ModelMode
import com.zeticai.mlange.core.model.QuantType
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.tensor.DataType
import com.zeticai.mlange.core.tensor.Tensor
import java.io.IOException
import java.net.UnknownHostException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.cert.CertPathValidatorException
import javax.net.ssl.SSLHandshakeException
import kotlin.math.exp

/**
 * Thin helpers around the ZeticMLange 1.8.1 Android API so each model file stays small.
 *
 * `ZeticMLangeModel(context, personalKey, name, version, modelMode, quantType, onProgress,
 *  cachePolicy)`; `fun run(inputs: Array<Tensor>): Array<Tensor>`.
 *
 * [ModelCacheHandlingPolicy.KEEP_EXISTING] keeps the on-device cache authoritative so a
 * downloaded model's bytes are reused without re-fetching. (Note: 1.8.x still performs an
 * online backend-selection handshake on each cold start, so the first launch per process
 * needs connectivity.)
 */
object MelangeKit {

    /** Load (and on first run, download + compile) a Melange model for this device's NPU. */
    fun load(
        context: Context,
        name: String,
        version: Int = 1,
        onProgress: (Float) -> Unit,
    ): ZeticMLangeModel =
        // 1.8.1's progress-callback constructor takes QuantType positionally (named
        // args don't resolve to it). FP32 keeps full precision, matching RUN_ACCURACY
        // and the ExecuTorch-FP32 emotion artifact.
        ZeticMLangeModel(
            context,
            AppConfig.PERSONAL_KEY,
            name,
            version,
            ModelMode.RUN_ACCURACY,
            QuantType.FP32,
            { progress -> onProgress(progress) },
            ModelCacheHandlingPolicy.KEEP_EXISTING,
        )

    /** True when [t] is a no-network failure (vs. a real model/runtime error). */
    fun isOffline(t: Throwable): Boolean {
        var c: Throwable? = t
        while (c != null) {
            if (c is UnknownHostException) return true
            if (c is IOException && (c.message?.contains("host", true) == true ||
                    c.message?.contains("Unable to resolve", true) == true)
            ) return true
            c = c.cause
        }
        return false
    }

    /**
     * True when [t] is a TLS trust failure (server cert chains to a CA the app doesn't trust —
     * typically a TLS-inspecting proxy/VPN whose root is a user-installed certificate). Distinct
     * from offline: the network is reachable, the handshake is rejected.
     */
    fun isTlsTrust(t: Throwable): Boolean {
        var c: Throwable? = t
        while (c != null) {
            if (c is CertPathValidatorException || c is SSLHandshakeException) return true
            if (c.message?.contains("Trust anchor", true) == true) return true
            c = c.cause
        }
        return false
    }

    /** UI-friendly text for a load/run failure. */
    fun friendly(t: Throwable): String =
        when {
            isOffline(t) ->
                "No internet. The model downloads once, then runs fully on-device — " +
                    "connect to Wi-Fi for the first launch, then tap Retry."
            isTlsTrust(t) ->
                // A captive portal (Wi-Fi sign-in page) or a TLS-inspecting proxy intercepts the
                // download and serves an untrusted certificate. Both produce this trust error.
                "Couldn't reach the model server securely. If your Wi-Fi has a sign-in page, sign " +
                    "in (or switch networks / disable any VPN-proxy), then tap Retry. Models " +
                    "download once, then run fully on-device."
            else -> t.message ?: t.toString()
        }

    /** Wrap a float array as a float32 input tensor with the given shape. */
    fun floatTensor(values: FloatArray, shape: IntArray): Tensor {
        val buffer = ByteBuffer
            .allocateDirect(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(values)
        buffer.rewind()
        return Tensor(buffer, DataType.Float32, shape)
    }

    /** Read raw tensor bytes back into a float array. */
    fun floats(tensor: Tensor): FloatArray {
        val buffer = tensor.data.order(ByteOrder.nativeOrder())
        buffer.rewind()
        val fb = buffer.asFloatBuffer()
        val out = FloatArray(fb.remaining())
        fb.get(out)
        return out
    }

    /** Number of float32 elements a tensor holds (used to identify YAMNet's score tensor). */
    fun floatCount(tensor: Tensor): Int = tensor.count()

    /**
     * Pad with zeros / trim (from the start) so a clip is exactly [count] samples,
     * to match a model's fixed NPU input length.
     */
    fun fit(samples: FloatArray, count: Int): FloatArray {
        if (samples.size == count) return samples
        if (samples.size > count) return samples.copyOfRange(0, count)
        return samples + FloatArray(count - samples.size)
    }

    /**
     * Fit to exactly [count] samples by **tiling** short clips (repeating the speech)
     * instead of zero-padding, and center-cropping long ones. Repeating avoids diluting
     * a mean-pooled model with silence frames — better for short emotion clips.
     */
    fun fitTiling(samples: FloatArray, count: Int): FloatArray {
        if (samples.isEmpty()) return FloatArray(count)
        if (samples.size == count) return samples
        if (samples.size > count) {
            val start = (samples.size - count) / 2          // centered crop
            return samples.copyOfRange(start, start + count)
        }
        val out = FloatArray(count)
        var i = 0
        while (i < count) {
            val n = minOf(samples.size, count - i)
            System.arraycopy(samples, 0, out, i, n)
            i += samples.size
        }
        return out
    }

    /** Numerically stable softmax. */
    fun softmax(x: FloatArray): FloatArray {
        if (x.isEmpty()) return x
        val m = x.max()
        val e = FloatArray(x.size) { exp(x[it] - m) }
        val sum = e.sum()
        return if (sum > 0f) FloatArray(x.size) { e[it] / sum } else e
    }
}

/** Measures wall-clock duration of a block, in milliseconds. */
inline fun <T> measureMs(body: () -> T): Pair<T, Double> {
    val start = System.nanoTime()
    val value = body()
    val ms = (System.nanoTime() - start) / 1_000_000.0
    return value to ms
}
