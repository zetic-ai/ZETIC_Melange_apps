package ai.zetic.skinclassifier.model

import androidx.compose.ui.graphics.Color
import ai.zetic.skinclassifier.ui.Theme

/** Clinical seriousness of a lesion class, driving tint + guidance copy. */
enum class Severity { BENIGN, PRECANCER, MALIGNANT }

/**
 * The 7 HAM10000 lesion classes the classifier emits, in **logit / id2label order**.
 *
 * The ordinal of each entry IS the model's output index — DO NOT reorder. It must equal the
 * model's `id2label` (verified against the iOS app's `SkinClass.allCases`). 1:1 with the iOS
 * `SkinClass` enum, including all curated guidance copy.
 */
enum class SkinClass(
    val title: String,
    val clinicalName: String,
    val severity: Severity,
    val blurb: String,
    val whatItMeans: String,
) {
    BENIGN_KERATOSIS(
        title = "Benign Keratosis",
        clinicalName = "benign keratosis-like lesion (e.g. seborrheic keratosis)",
        severity = Severity.BENIGN,
        blurb = "A common non-cancerous skin growth.",
        whatItMeans = "Benign keratoses (like seborrheic keratoses) are common, harmless skin " +
            "growths that often appear with age. They can look waxy or 'stuck on' and are not " +
            "cancerous.",
    ),
    BASAL_CELL(
        title = "Basal Cell Carcinoma",
        clinicalName = "basal cell carcinoma",
        severity = Severity.MALIGNANT,
        blurb = "The most common form of skin cancer; usually slow-growing.",
        whatItMeans = "Basal cell carcinoma is the most common skin cancer. It tends to grow " +
            "slowly and rarely spreads, but it should be evaluated and treated by a clinician.",
    ),
    ACTINIC_KERATOSES(
        title = "Actinic Keratosis",
        clinicalName = "actinic keratosis",
        severity = Severity.PRECANCER,
        blurb = "Sun-damage spot that can be pre-cancerous.",
        whatItMeans = "Actinic keratoses are rough, scaly patches caused by long-term sun " +
            "exposure. They are considered pre-cancerous because a small fraction can progress " +
            "over time.",
    ),
    VASCULAR(
        title = "Vascular Lesion",
        clinicalName = "vascular lesion (e.g. angioma)",
        severity = Severity.BENIGN,
        blurb = "A benign growth of blood vessels.",
        whatItMeans = "Vascular lesions, such as cherry angiomas, are benign overgrowths of " +
            "small blood vessels. They are very common and generally harmless.",
    ),
    MELANOCYTIC_NEVI(
        title = "Melanocytic Nevus",
        clinicalName = "melanocytic nevus (common mole)",
        severity = Severity.BENIGN,
        blurb = "An ordinary mole — typically harmless.",
        whatItMeans = "Melanocytic nevi are ordinary moles — clusters of pigment cells. The " +
            "vast majority are harmless, but moles that change warrant a check.",
    ),
    MELANOMA(
        title = "Melanoma",
        clinicalName = "melanoma",
        severity = Severity.MALIGNANT,
        blurb = "A serious skin cancer that needs prompt attention.",
        whatItMeans = "Melanoma is a serious form of skin cancer that can spread if not caught " +
            "early. Early evaluation greatly improves outcomes.",
    ),
    DERMATOFIBROMA(
        title = "Dermatofibroma",
        clinicalName = "dermatofibroma",
        severity = Severity.BENIGN,
        blurb = "A benign firm nodule, often on the legs.",
        whatItMeans = "Dermatofibroma is a common, benign firm bump, often on the legs. It is " +
            "harmless and usually needs no treatment.",
    );

    /** Badge text shown for the lesion's severity. */
    val severityLabel: String
        get() = when (severity) {
            Severity.BENIGN -> "Typically benign"
            Severity.PRECANCER -> "Pre-cancerous"
            Severity.MALIGNANT -> "Potentially serious"
        }

    /** Accent tint for the verdict / ring / bars. */
    val tint: Color
        get() = when (severity) {
            Severity.BENIGN -> Theme.Mint
            Severity.PRECANCER -> Theme.Amber
            Severity.MALIGNANT -> Theme.Coral
        }

    /** General self-care bullets, keyed by severity (matches iOS). */
    val selfCare: List<String>
        get() = when (severity) {
            Severity.MALIGNANT, Severity.PRECANCER -> listOf(
                "Avoid scratching, picking, or irritating the area.",
                "Protect the skin from the sun (SPF 30+, cover up).",
                "Photograph it so you can track any changes over time.",
            )
            Severity.BENIGN -> listOf(
                "Usually no treatment is needed.",
                "Protect your skin from the sun to keep it healthy.",
                "Note the spot's size and color so you'd notice a change.",
            )
        }

    /** When-to-see-a-doctor copy, keyed by severity (matches iOS). */
    val whenToSeeDoctor: String
        get() = when (severity) {
            Severity.MALIGNANT ->
                "See a dermatologist promptly to have this evaluated. Don't wait if it grows, " +
                    "bleeds, or changes."
            Severity.PRECANCER ->
                "Have a clinician check this within the next few weeks, sooner if it changes."
            Severity.BENIGN ->
                "Routine — but see a clinician if it changes shape, color, size, itches, or " +
                    "bleeds (the ABCDE warning signs)."
        }

    companion object {
        /** Indexed by model logit position. */
        val ordered: List<SkinClass> = entries.toList()
    }
}
