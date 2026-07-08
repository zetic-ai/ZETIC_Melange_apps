import Foundation

/// Central configuration for the **Skin Image Classification** demo — ZETIC Melange's on-device
/// AI shown to skin-image AI companies.
///
/// The whole pitch in one screen: a client's **own** dermatology vision model and a
/// **medical LLM** run *together, fully on-device* through Melange. No pixels and no
/// text ever leave the phone. Swap either `name` below for a client's Melange model
/// and the rest of the app keeps working unchanged.
///
/// Pipeline:
///   photo → `SkinClassifier` (ViT, 7 classes) → top class + confidence
///         → `MedGemmaService` (MedGemma-4b text LLM) → plain-language explanation
enum AppConfig {

    /// ZETIC Melange Personal Access Key. `adapt_mlange_key.sh` (run from the repo root)
    /// replaces this placeholder with your token. Get yours at
    /// https://mlange.zetic.ai → Settings.
    static let personalKey = "YOUR_MLANGE_KEY"

    // MARK: - Models (already deployed on the Melange dashboard)

    enum Model {
        /// Skin-lesion image classifier — `Anwarkh1/Skin_Cancer-Image_Classification`
        /// (ViT-base-patch16-224, 7 HAM10000 classes, ~97% val acc), deployed to Melange.
        /// Input `1×3×224×224` NCHW Float32, RGB, normalized to [-1, 1]. Output: 7 logits.
        static let classifier = "realtonypark/Skin_Cancer-Image_Classification"
        static let classifierVersion = 1

        /// MedGemma 4b-it — Google's medical Gemma, deployed to Melange. Text-only LLM
        /// (the SDK has no image input), so it reasons over the classifier's *result*.
        static let medGemma = "Steve/Medgemma-1.5-4b-it"
        static let medGemmaVersion = 1
    }

    // MARK: - Classifier preprocessing

    /// Tunable so the on-device tensor layout can be matched to whatever the
    /// Melange-converted graph expects (NCHW vs NHWC, RGB vs BGR, normalization).
    /// Defaults are the PyTorch ViT contract; verify on-device against the Python
    /// oracle (see README) and flip these if the argmax disagrees.
    enum Preprocess {
        static let inputSize = 224

        enum Layout { case nchw, nhwc }
        enum ChannelOrder { case rgb, bgr }
        /// `signed1`: (px/255 − 0.5)/0.5 → [-1, 1]  (ViT default, mean=std=0.5)
        /// `unit`:    px/255 → [0, 1]
        enum Normalize { case signed1, unit }

        // Confirmed against model_conversion/convert.py + output/labels.json:
        // the exported .pt2 takes NCHW (1,3,224,224) float32 and returns (1,7) logits,
        // normalized by HuggingFace ViTImageProcessor (mean=std=0.5 → [-1,1]).
        static let layout: Layout = .nchw
        static let channelOrder: ChannelOrder = .rgb
        static let normalize: Normalize = .signed1
        /// The HF ViTImageProcessor does a plain resize (stretch) to 224×224 with no
        /// center crop, so we match it. Flip to `true` only if on-device validation
        /// shows center-cropping improves real-world phone photos.
        static let centerCropSquare = false
    }

    // MARK: - LLM generation

    enum LLM {
        /// Context window. Our prompt + ~180-word answer fit comfortably in 2048;
        /// keeping it small reduces KV-cache memory and prefill latency.
        static let contextTokens = 2048
        /// Hard cap on generated tokens per analysis.
        static let maxTokens = 420
    }

    /// Diagnostic switches — set false for shipping demos.
    enum Debug {
        /// On launch (once the classifier is ready) run inference on a generated test
        /// image and log each step, so the vision path can be verified without a tap.
        static let selfTestClassifierOnLaunch = false
    }
}
