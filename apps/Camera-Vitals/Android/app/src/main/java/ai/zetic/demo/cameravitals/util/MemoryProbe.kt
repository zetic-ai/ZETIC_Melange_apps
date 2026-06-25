package ai.zetic.demo.cameravitals.util

import android.os.Debug
import android.util.Log

/** Logs the process memory footprint, for confirming memory plateaus on device. */
object MemoryProbe {
    fun footprintMB(): Double {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024.0   // PSS reported in KB → MB
    }

    fun log(tag: String) {
        Log.d("mem", "%-16s %.0f MB".format(tag, footprintMB()))
    }
}
