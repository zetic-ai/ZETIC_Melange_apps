package ai.zetic.demo.cherrypad.model

/** Rewrite tones. `descriptor` is injected into the rewrite prompt. */
enum class Tone(val label: String, val descriptor: String) {
    PROFESSIONAL("Professional", "polished and professional"),
    CASUAL("Casual", "relaxed and casual"),
    FRIENDLY("Friendly", "warm and friendly"),
    ROMANTIC("Romantic", "affectionate and romantic"),
}
