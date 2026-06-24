package ai.zetic.aiberry.ui

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

/**
 * Tiny haptics helper mirroring iOS-Aiberry's `Haptics` (light tap on primary actions,
 * a success buzz when the report is ready). No-ops gracefully if the device has no vibrator.
 */
object Haptics {
    private fun vibrator(context: Context): Vibrator? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val mgr = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
            mgr?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }

    fun tap(context: Context) {
        val v = vibrator(context)?.takeIf { it.hasVibrator() } ?: return
        // Never let a missing VIBRATE permission / OEM quirk crash a button tap.
        runCatching { v.vibrate(VibrationEffect.createOneShot(12, VibrationEffect.DEFAULT_AMPLITUDE)) }
    }

    fun success(context: Context) {
        val v = vibrator(context)?.takeIf { it.hasVibrator() } ?: return
        runCatching { v.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 18, 60, 28), -1)) }
    }
}
