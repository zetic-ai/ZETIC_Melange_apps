package ai.zetic.skinclassifier.core

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Shared serial executor for **every** ZeticMLange operation (download / compile / run).
 *
 * The SDK does not support concurrent model init — and racing two inits also corrupts its
 * network client, which surfaces as spurious `UnknownHostException`s. Funnelling all work
 * onto a single thread guarantees init and run never overlap.
 */
object MelangeRuntime {
    val executor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "melange-runtime").apply { isDaemon = true }
    }
}
