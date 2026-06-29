package ai.zetic.skinclassifier.core

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
 * Thin helpers around the ZeticMLange 1.8.1 Android API so the classifier stays small.
 *
 * Quick-start usage is `ZeticMLangeModel(context, key, name, version, modelMode = RUN_AUTO,
 * onProgress = { p -> })`, but 1.8.1's progress-callback constructor has no ModelMode+onProgress
 * overload — onProgress must be preceded by a QuantType and followed by a cache policy. So the
 * call below is the same intent (RUN_AUTO + progress) in 1.8.1's required positional shape.
 * `fun run(inputs: Array<Tensor>): Array<Tensor>`.
 */
object MelangeKit {

    /** Load (and on first run, download + compile) the Melange model for this device's NPU. */
    fun load(
        context: Context,
        name: String,
        version: Int,
        onProgress: (Float) -> Unit,
    ): ZeticMLangeModel =
        ZeticMLangeModel(
            context,
            AppConfig.PERSONAL_KEY,
            name,
            version,
            ModelMode.RUN_AUTO,
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

    /** True when [t] is a TLS trust failure (TLS-inspecting proxy / captive portal). */
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
                    "connect to Wi-Fi for the first launch, then tap Try again."
            isTlsTrust(t) ->
                "Couldn't reach the model server securely. If your Wi-Fi has a sign-in page, " +
                    "sign in (or switch networks / disable any VPN-proxy), then tap Try again. " +
                    "Models download once, then run fully on-device."
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
