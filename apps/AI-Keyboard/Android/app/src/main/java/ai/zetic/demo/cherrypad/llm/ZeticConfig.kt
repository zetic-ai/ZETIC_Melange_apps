package ai.zetic.demo.cherrypad.llm

import com.zeticai.mlange.core.model.llm.LLMModelMode

/**
 * On-device model configuration. LFM2.5-350M (Liquid Foundation Model) is a small
 * non-reasoning instruct model (~0.3 GB) that powers all four AI actions.
 *
 * The [PERSONAL_KEY] is the ZETIC.ai Melange dev token. Keep it here so it can be
 * swapped for a per-build key if desired.
 */
object ZeticConfig {
    const val PERSONAL_KEY = "dev_40e3948ba051485c9ccd827a2a17922f"
    const val MODEL_NAME = "Steve/LFM2.5_350M"
    const val MODEL_VERSION = 1
    val MODEL_MODE: LLMModelMode = LLMModelMode.RUN_AUTO
}
