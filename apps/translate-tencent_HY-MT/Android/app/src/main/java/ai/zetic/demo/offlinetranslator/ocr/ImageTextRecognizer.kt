package ai.zetic.demo.offlinetranslator.ocr

import android.content.Context
import android.net.Uri
import android.util.Log
import ai.zetic.demo.offlinetranslator.model.Language
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

/**
 * Offline OCR via ML Kit Text Recognition (bundled models). The script model is chosen from the
 * selected source language; Latin is the default. Scripts ML Kit doesn't cover (Arabic, Thai,
 * Hebrew, …) fall back to Latin — acceptable for this demo. Runs ML Kit's own worker; success/
 * failure callbacks arrive on the main thread.
 */
object ImageTextRecognizer {

    private fun recognizerFor(language: Language): TextRecognizer = when (language.id) {
        "zh-Hans", "zh-Hant" -> TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())
        "ja" -> TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build())
        "ko" -> TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
        "hi" -> TextRecognition.getClient(DevanagariTextRecognizerOptions.Builder().build())
        else -> TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }

    fun recognize(
        context: Context,
        uri: Uri,
        language: Language,
        onResult: (String) -> Unit,
        onError: (String) -> Unit,
    ) {
        val image = try {
            InputImage.fromFilePath(context, uri) // handles EXIF rotation
        } catch (t: Throwable) {
            onError(t.message ?: "Couldn't read the image.")
            return
        }
        val recognizer = recognizerFor(language)
        recognizer.process(image)
            .addOnSuccessListener { text ->
                onResult(text.text)
                recognizer.close()
            }
            .addOnFailureListener { e ->
                Log.e("ImageOCR", "OCR failed", e)
                onError(e.message ?: "Couldn't recognize text.")
                recognizer.close()
            }
    }
}
