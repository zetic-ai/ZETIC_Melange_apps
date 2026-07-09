package ai.zetic.demo.posemotion.benchmark

import android.os.Debug
import android.util.Log

/** Process PSS in MB — for confirming the memory plateau on device. */
object MemoryProbe {
    fun footprintMB(): Double {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024.0
    }

    fun log(tag: String) {
        Log.d("mem", "[mem] %-24s %.0f MB".format(tag, footprintMB()))
    }
}
