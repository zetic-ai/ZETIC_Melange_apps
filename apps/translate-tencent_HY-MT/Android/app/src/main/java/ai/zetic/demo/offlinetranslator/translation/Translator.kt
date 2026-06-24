package ai.zetic.demo.offlinetranslator.translation

import android.content.Context
import android.os.Build
import android.util.Log
import ai.zetic.demo.offlinetranslator.model.Language

/**
 * Abstraction over the on-device translation engine. `load` and `generate` are **blocking** —
 * the ZeticMLange implementation calls native code that must run on a single dedicated thread,
 * never a thread pool. The caller (the ViewModel) owns the off-main execution. Mirrors the iOS
 * `Translator` protocol.
 */
interface Translator {
    /** Prepare the model, reporting download progress in 0..1. Throws on failure. */
    fun load(onProgress: (Double) -> Unit)

    /**
     * Run one generation for [prompt], invoking [onToken] for each produced token.
     * [onToken] returns `false` to request an early stop (e.g. on cancellation).
     */
    fun generate(prompt: String, onToken: (String) -> Boolean)

    /** Reset conversation / KV-cache state between translations. */
    fun reset()

    /** Fully release native resources. */
    fun tearDown()
}

class TranslatorNotLoadedException :
    IllegalStateException("The translation model isn't ready yet.")

/**
 * Chooses the real SDK engine on capable hardware, else a mock — the runtime equivalent of iOS's
 * compile-time `#if canImport(ZeticMLange)` split across two targets. The ZeticMLange native
 * libraries are arm64-only, so x86_64 emulators (and any device where native init fails) fall
 * back to [MockTranslator] so the full UI stays demoable.
 */
object TranslatorFactory {
    private const val TAG = "TranslatorFactory"

    fun create(context: Context): Translator {
        if (!Build.SUPPORTED_ABIS.contains("arm64-v8a")) {
            Log.i(TAG, "No arm64-v8a ABI; using MockTranslator for UI.")
            return MockTranslator()
        }
        return try {
            ZeticTranslator(context.applicationContext)
        } catch (t: Throwable) {
            Log.w(TAG, "ZeticMLange unavailable (${t.javaClass.simpleName}); falling back to mock.", t)
            MockTranslator()
        }
    }
}

/**
 * Builds the prompt in Tencent **Hunyuan-MT**'s official instruction template. The model was
 * trained on these exact phrasings; a generic "Translate … to X" prompt makes it unreliable (it
 * sometimes echoes the source language instead of translating). Only the *target* is specified —
 * Hunyuan infers the source — so an explicit source selection is a UI affordance, not part of the
 * prompt. Verbatim from iOS `Translation/Translator.swift`.
 */
object TranslationPrompt {
    fun make(text: String, @Suppress("UNUSED_PARAMETER") from: Language, to: Language): String {
        val body = text.trim()
        if (to.id == "zh-Hans" || to.id == "zh-Hant") {
            // Hunyuan-MT Chinese template, used when translating into Chinese (ZH<=>XX).
            val name = if (to.id == "zh-Hant") "繁体中文" else "简体中文"
            return "把下面的文本翻译成${name}，不要额外解释。\n\n$body"
        }
        // Hunyuan-MT English template for every other target (XX<=>XX).
        val name = if (to.id == "en") "English" else to.englishName
        return "Translate the following segment into $name, without additional explanation.\n\n$body"
    }
}

/**
 * Trims artifacts from streamed model output (leading whitespace, an accidental echo of the
 * prompt's lead-in). Verbatim logic from iOS `TranslationCleanup`.
 */
object TranslationCleanup {
    private val markers = listOf(
        "Translate the following segment",
        "Translate the following text",
        "把下面的文本翻译成",
    )

    fun clean(raw: String): String {
        var text = raw.trim()
        // If the model echoed the instruction line, drop it (Hunyuan rarely does, but be safe).
        for (marker in markers) {
            if (text.startsWith(marker)) {
                val newline = text.indexOf('\n')
                if (newline >= 0) text = text.substring(newline + 1).trim()
            }
        }
        return text
    }
}
