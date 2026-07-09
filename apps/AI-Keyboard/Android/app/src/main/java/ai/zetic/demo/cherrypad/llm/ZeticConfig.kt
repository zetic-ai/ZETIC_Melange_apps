package ai.zetic.demo.cherrypad.llm

import com.zeticai.mlange.core.model.llm.LLMModelMode

/**
 * On-device model configuration. LFM2.5-350M (Liquid Foundation Model) is a small
 * non-reasoning instruct model (~0.3 GB) that powers all four AI actions.
 *
 * [PERSONAL_KEY] is the placeholder Melange Personal Access Token — run
 * `./adapt_mlange_key.sh` from the repo root to inject the real key (keeps it out of git).
 */
object ZeticConfig {
    const val PERSONAL_KEY = "YOUR_MLANGE_KEY"
    const val MODEL_NAME = "Steve/LFM2.5_350M"
    const val MODEL_VERSION = 1
    val MODEL_MODE: LLMModelMode = LLMModelMode.RUN_AUTO
}
