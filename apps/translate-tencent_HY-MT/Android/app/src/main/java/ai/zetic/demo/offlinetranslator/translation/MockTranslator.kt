package ai.zetic.demo.offlinetranslator.translation

/**
 * Mock engine for emulators / previews where the arm64 ZeticMLange native library can't load.
 * Streams a canned translation token-by-token with a simulated download ramp so every screen and
 * the streaming caret are exercisable without the real SDK. Mirrors iOS `MockTranslator`.
 */
class MockTranslator : Translator {

    override fun load(onProgress: (Double) -> Unit) {
        // Simulate a brief one-time "download" so the loading overlay is demoable.
        var p = 0.0
        while (p < 1.0) {
            p = (p + 0.08).coerceAtMost(1.0)
            onProgress(p)
            sleep(60)
        }
    }

    override fun generate(prompt: String, onToken: (String) -> Boolean) {
        // A plausible canned result, streamed word-by-word.
        val canned = "Deploy your own AI model locally on any device using Zetic."
        sleep(250) // brief "thinking" pause so the spinner shows first
        for (word in canned.split(" ")) {
            if (!onToken("$word ")) return
            sleep(45)
        }
    }

    override fun reset() {}

    override fun tearDown() {}

    private fun sleep(ms: Long) {
        try {
            Thread.sleep(ms)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }
}
