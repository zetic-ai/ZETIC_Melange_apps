package ai.zetic.demo.cherrypad.model

/** Reply stances. `descriptor` is injected into the reply prompt. */
enum class Stance(val label: String, val descriptor: String) {
    AGREEABLE("Agreeable", "agreeable and supportive"),
    DISAGREEABLE("Disagreeable", "politely disagreeing"),
}
