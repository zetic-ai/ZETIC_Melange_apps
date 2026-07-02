package ai.zetic.demo.cherrypad.model

/**
 * The four AI actions. `title` is the button label; `tagline` mirrors the iOS copy
 * (used in marketing/onboarding, not shown on the action buttons). Declaration order
 * — rewrite, reply, translate, grammar — drives the on-screen order.
 */
enum class KeyboardTask(val title: String, val tagline: String) {
    REWRITE("Rewrite", "Better words, same you."),
    REPLY("Reply", "Let it reply, reduce mental load."),
    TRANSLATE("Translate", "Cross language barriers."),
    GRAMMAR("Grammar", "Fix grammar, keep your tone."),
}
