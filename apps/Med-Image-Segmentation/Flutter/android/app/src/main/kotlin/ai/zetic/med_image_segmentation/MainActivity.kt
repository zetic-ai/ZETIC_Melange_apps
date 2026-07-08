package ai.zetic.med_image_segmentation

import android.os.Debug
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    // medseg/memory — the plugin exposes no memory API, so report totalPss for the
    // benchmark's peak-memory column.
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "medseg/memory")
            .setMethodCallHandler { call, reply ->
                if (call.method == "footprintMB") {
                    val mem = Debug.MemoryInfo()
                    Debug.getMemoryInfo(mem)
                    reply.success(mem.totalPss / 1024.0) // KB -> MB
                } else {
                    reply.notImplemented()
                }
            }
    }
}
