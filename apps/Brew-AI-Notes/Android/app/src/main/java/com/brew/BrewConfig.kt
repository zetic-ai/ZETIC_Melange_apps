package com.brew

/**
 * Central configuration for the ZeticMLange on-device models.
 *
 * The personal access key is embedded here (matching the repo's other ZeticMLange demos). For a
 * production app prefer a `buildConfigField` sourced from `local.properties`.
 */
object BrewConfig {
    const val PERSONAL_KEY = "dev_40e3948ba051485c9ccd827a2a17922f"

    // Instruction-tuned LLM — note enhancement + grounded chat. Direct-download
    // registered (works on sideloaded builds, unlike Play-Asset-Delivery models).
    const val LLM_MODEL = "SJ_zetic/Ministral-3-3B-Instruct-2512"
    const val LLM_VERSION = 1

    // How many times to retry a model download/load before surfacing a failure.
    const val MODEL_LOAD_ATTEMPTS = 3

    // Whisper-small encoder/decoder split — on-device speech-to-text.
    const val WHISPER_ENCODER_MODEL = "vaibhav-zetic/whisper_small_encoder"
    const val WHISPER_ENCODER_VERSION = 3
    const val WHISPER_DECODER_MODEL = "vaibhav-zetic/whisper_small_decoder"
    const val WHISPER_DECODER_VERSION = 1
}
