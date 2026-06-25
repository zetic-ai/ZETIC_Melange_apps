package ai.zetic.aiberry.emotion

import androidx.compose.ui.graphics.Color
import ai.zetic.aiberry.ui.Theme

/** Visual identity (emoji + color) for each emotion label, for a polished result UI. */
object EmotionStyle {
    fun emoji(label: String): String = when (label) {
        "Angry" -> "😠"      // 😠
        "Disgust" -> "🤢"    // 🤢
        "Fear" -> "😨"       // 😨
        "Happy" -> "😄"      // 😄
        "Neutral" -> "😐"    // 😐
        "Sad" -> "😢"        // 😢
        "Surprise" -> "😲"   // 😲
        else -> "🎙️"   // 🎙️
    }

    fun color(label: String): Color = when (label) {
        "Angry" -> Color(0.90f, 0.30f, 0.30f)
        "Disgust" -> Color(0.45f, 0.65f, 0.30f)
        "Fear" -> Color(0.55f, 0.40f, 0.85f)
        "Happy" -> Color(0.97f, 0.74f, 0.20f)
        "Neutral" -> Color(0.55f, 0.58f, 0.62f)
        "Sad" -> Color(0.30f, 0.55f, 0.90f)
        "Surprise" -> Color(0.95f, 0.55f, 0.25f)
        else -> Theme.Accent
    }
}
