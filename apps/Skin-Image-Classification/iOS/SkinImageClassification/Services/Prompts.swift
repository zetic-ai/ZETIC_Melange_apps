import Foundation

/// Builds the MedGemma prompt from the classifier's result.
///
/// Guardrails baked into the prompt (MedGemma is text-only — it never sees the image):
///  • It explains the *classifier's* output; it must not re-diagnose or invent numbers.
///  • Tone tracks the given confidence and the lesion's severity.
///  • Output is constrained to three fixed Markdown sections + a disclaimer line, so
///    `ResultsView` renders predictably and the model can't ramble into a diagnosis.
enum Prompts {

    static func explanation(for c: Classification) -> String {
        let pct = Int((c.confidence * 100).rounded())
        let top = c.topClass

        // Ranked distribution as a compact, readable list.
        let distribution = c.ranked
            .prefix(4)
            .map { "\($0.skinClass.title): \(Int(($0.probability * 100).rounded()))%" }
            .joined(separator: ", ")

        var rules = [
            "If confidence is below 60%, clearly say the result is uncertain and should not be relied on."
        ]
        switch top.severity {
        case .malignant:
            rules.append("This category can be a skin cancer — calmly but clearly recommend seeing a dermatologist promptly. Do not reassure the reader that it is harmless.")
        case .precancer:
            rules.append("This category can be pre-cancerous — recommend getting it checked by a clinician.")
        case .benign:
            rules.append("This category is usually benign — reassure appropriately, while noting that any changing, bleeding, or new lesion still warrants a professional check.")
        }
        let ruleText = rules.map { "- \($0)" }.joined(separator: "\n")

        return """
        You are a medical information assistant inside an on-device skin-screening app. \
        An image classifier (not you) has analyzed a photo of a skin lesion. You CANNOT \
        see the image. Do NOT diagnose, do NOT claim certainty, and do NOT invent any \
        probabilities beyond those provided. Write for a worried non-expert in plain, \
        calm language, about 160 words total.

        Output EXACTLY these three Markdown sections, in this order, and nothing before or after them except the final disclaimer line:
        ## What this result may suggest
        ## General self-care
        ## When to seek medical care

        End with exactly this line:
        _This is not a medical diagnosis. Please consult a healthcare professional._

        Classifier result:
        - Most likely category: \(top.title) (\(top.clinicalName)) — confidence \(pct)%
        - Full distribution: \(distribution)

        Guidance:
        \(ruleText)
        """
    }
}
