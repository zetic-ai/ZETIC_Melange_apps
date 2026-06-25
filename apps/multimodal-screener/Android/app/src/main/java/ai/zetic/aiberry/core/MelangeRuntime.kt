package ai.zetic.aiberry.core

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Shared serial executor for **every** ZeticMLange operation across all models.
 *
 * The SDK does not support concurrent model init ("Concurrent model init is not
 * supported yet") — and racing two inits also corrupts its network client, which
 * surfaces as spurious `UnknownHostException`s. Funnelling all download / compile /
 * inference work onto a single thread guarantees init and run never overlap.
 */
object MelangeRuntime {
    val executor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "melange-runtime").apply { isDaemon = true }
    }
}
