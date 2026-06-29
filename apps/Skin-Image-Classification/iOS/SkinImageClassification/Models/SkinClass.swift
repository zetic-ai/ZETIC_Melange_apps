import SwiftUI

/// The 7 HAM10000 lesion categories the on-device classifier predicts.
///
/// `allCases` order **must match the model's logit order** (id2label). This is the
/// `Anwarkh1/Skin_Cancer-Image_Classification` order; verify on deploy and reorder
/// here if the export reorders the head (see README "Validate the classifier").
enum SkinClass: Int, CaseIterable, Identifiable {
    case benignKeratosis = 0   // Benign keratosis-like lesions
    case basalCell             // Basal cell carcinoma
    case actinicKeratoses      // Actinic keratoses
    case vascular              // Vascular lesions
    case melanocyticNevi       // Melanocytic nevi
    case melanoma              // Melanoma
    case dermatofibroma        // Dermatofibroma

    var id: Int { rawValue }

    /// Short, lay-friendly display name.
    var title: String {
        switch self {
        case .benignKeratosis: return "Benign Keratosis"
        case .basalCell:       return "Basal Cell Carcinoma"
        case .actinicKeratoses: return "Actinic Keratosis"
        case .vascular:        return "Vascular Lesion"
        case .melanocyticNevi: return "Melanocytic Nevus"
        case .melanoma:        return "Melanoma"
        case .dermatofibroma:  return "Dermatofibroma"
        }
    }

    /// The clinical label name, passed to MedGemma so it reasons about the right entity.
    var clinicalName: String {
        switch self {
        case .benignKeratosis: return "benign keratosis-like lesion (e.g. seborrheic keratosis)"
        case .basalCell:       return "basal cell carcinoma"
        case .actinicKeratoses: return "actinic keratosis"
        case .vascular:        return "vascular lesion (e.g. angioma)"
        case .melanocyticNevi: return "melanocytic nevus (common mole)"
        case .melanoma:        return "melanoma"
        case .dermatofibroma:  return "dermatofibroma"
        }
    }

    /// One-line plain description shown under the class name.
    var blurb: String {
        switch self {
        case .benignKeratosis: return "A common non-cancerous skin growth."
        case .basalCell:       return "The most common form of skin cancer; usually slow-growing."
        case .actinicKeratoses: return "Sun-damage spot that can be pre-cancerous."
        case .vascular:        return "A benign growth of blood vessels."
        case .melanocyticNevi: return "An ordinary mole — typically harmless."
        case .melanoma:        return "A serious skin cancer that needs prompt attention."
        case .dermatofibroma:  return "A benign firm nodule, often on the legs."
        }
    }

    /// How concerning the category is — drives UI accent and MedGemma's triage tone.
    enum Severity { case benign, precancer, malignant }

    var severity: Severity {
        switch self {
        case .melanoma, .basalCell: return .malignant
        case .actinicKeratoses:     return .precancer
        case .benignKeratosis, .vascular, .melanocyticNevi, .dermatofibroma: return .benign
        }
    }

    /// Accent color for the result card / bars.
    var tint: Color {
        switch severity {
        case .benign:    return Theme.mint
        case .precancer: return Theme.amber
        case .malignant: return Theme.coral
        }
    }

    /// Short risk word for the badge.
    var severityLabel: String {
        switch severity {
        case .benign:    return "Typically benign"
        case .precancer: return "Pre-cancerous"
        case .malignant: return "Potentially serious"
        }
    }

    // MARK: - Curated guidance (static, reviewed copy — not AI-generated)

    /// Plain-language "what this category is" for the result screen.
    var whatItMeans: String {
        switch self {
        case .benignKeratosis:
            return "Benign keratoses (like seborrheic keratoses) are common, harmless skin growths that often appear with age. They can look waxy or 'stuck on' and are not cancerous."
        case .basalCell:
            return "Basal cell carcinoma is the most common skin cancer. It tends to grow slowly and rarely spreads, but it should be evaluated and treated by a clinician."
        case .actinicKeratoses:
            return "Actinic keratoses are rough, scaly patches caused by long-term sun exposure. They are considered pre-cancerous because a small fraction can progress over time."
        case .vascular:
            return "Vascular lesions, such as cherry angiomas, are benign overgrowths of small blood vessels. They are very common and generally harmless."
        case .melanocyticNevi:
            return "Melanocytic nevi are ordinary moles — clusters of pigment cells. The vast majority are harmless, but moles that change warrant a check."
        case .melanoma:
            return "Melanoma is a serious form of skin cancer that can spread if not caught early. Early evaluation greatly improves outcomes."
        case .dermatofibroma:
            return "Dermatofibroma is a common, benign firm bump, often on the legs. It is harmless and usually needs no treatment."
        }
    }

    /// General, non-prescriptive self-care guidance.
    var selfCare: [String] {
        switch severity {
        case .malignant, .precancer:
            return [
                "Avoid scratching, picking, or irritating the area.",
                "Protect the skin from the sun (SPF 30+, cover up).",
                "Photograph it so you can track any changes over time."
            ]
        case .benign:
            return [
                "Usually no treatment is needed.",
                "Protect your skin from the sun to keep it healthy.",
                "Note the spot's size and color so you'd notice a change."
            ]
        }
    }

    /// When to seek professional care.
    var whenToSeeDoctor: String {
        switch severity {
        case .malignant:
            return "See a dermatologist promptly to have this evaluated. Don't wait if it grows, bleeds, or changes."
        case .precancer:
            return "Have a clinician check this within the next few weeks, sooner if it changes."
        case .benign:
            return "Routine — but see a clinician if it changes shape, color, size, itches, or bleeds (the ABCDE warning signs)."
        }
    }
}
