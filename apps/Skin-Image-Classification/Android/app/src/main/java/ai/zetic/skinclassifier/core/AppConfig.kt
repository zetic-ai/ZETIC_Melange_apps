package ai.zetic.skinclassifier.core

/**
 * Central configuration for the DermaScope on-device skin-classification demo.
 *
 * A 1:1 port of the iOS app's `AppConfig.swift`: a single ViT image classifier
 * (`realtonypark/Skin_Cancer-Image_Classification`, 7 HAM10000 lesion types) deployed on
 * ZETIC Melange and run fully on-device. Swap [Model.CLASSIFIER] / [Model.VERSION] for your
 * own Melange model (and the labels in [ai.zetic.skinclassifier.model.SkinClass]) to demo it.
 */
object AppConfig {
    /** ZETIC Melange Personal Access Key (shared dev key for this demo).
     *  Replace with your own from https://mlange.zetic.ai -> Settings. */
    const val PERSONAL_KEY = "dev_40e3948ba051485c9ccd827a2a17922f"

    /** Melange model identifier (already hosted). */
    object Model {
        /** ViT-base skin-lesion classifier, 7 classes, version 1. */
        const val CLASSIFIER = "realtonypark/Skin_Cancer-Image_Classification"
        const val VERSION = 1
    }

    /**
     * Input contract for the classifier. The PyTorch ViT expects NCHW `[1,3,224,224]`, RGB,
     * normalized to `[-1,1]` (mean=std=0.5 -> `px/127.5 - 1`). If the Melange-converted graph
     * expects a different layout/channel order/normalization, flip the flags here — the #1
     * source of a "wrong prediction on device" bug.
     */
    object Preprocess {
        const val INPUT_SIZE = 224

        enum class Layout { NCHW, NHWC }
        enum class ChannelOrder { RGB, BGR }
        enum class Normalize { SIGNED1, UNIT } // [-1,1] vs [0,1]

        val layout = Layout.NCHW
        val channelOrder = ChannelOrder.RGB
        val normalize = Normalize.SIGNED1
    }

    /** Below this top-class probability the result is flagged "low confidence". */
    const val LOW_CONFIDENCE = 0.60f
}
