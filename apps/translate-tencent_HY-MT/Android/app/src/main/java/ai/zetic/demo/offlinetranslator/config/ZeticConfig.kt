package ai.zetic.demo.offlinetranslator.config

/**
 * ZETIC.ai Melange deployment config for the on-device translation model.
 *
 * Note: the iOS app uses the slug `vaibhav-zetic/tencent_HY-MT`; this Android build uses
 * `palm/tencent_HY-MT` per the provided Android deployment snippet (same dev key, version 1).
 * If the model fails to download on a real device, this slug is the first thing to verify.
 */
object ZeticConfig {
    const val personalKey = "YOUR_MLANGE_KEY"
    const val modelName = "palm/tencent_HY-MT"
    const val modelVersion = 1
}
