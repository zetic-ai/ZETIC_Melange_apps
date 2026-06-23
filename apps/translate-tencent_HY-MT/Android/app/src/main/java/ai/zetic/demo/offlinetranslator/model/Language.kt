package ai.zetic.demo.offlinetranslator.model

import java.util.Locale

/**
 * A translation language. `auto` is a source-only sentinel meaning "let the model infer the
 * source language" — the prompt only ever needs the *target*, so Detect needs no special handling.
 * Mirrors `Models/Language.swift`.
 */
data class Language(
    val id: String,          // stable identifier / rough BCP-47 code
    val englishName: String, // name used when building the model prompt
    val nativeName: String,  // endonym shown in the picker
) {
    val isDetect: Boolean get() = id == "auto"

    /** BCP-47-ish locale for Android TextToSpeech voice selection. */
    val speechLocale: Locale
        get() = when (id) {
            "auto", "en" -> Locale.US
            "zh-Hans" -> Locale.SIMPLIFIED_CHINESE
            "zh-Hant" -> Locale.TRADITIONAL_CHINESE
            "nb" -> Locale("nb", "NO")
            else -> Locale.forLanguageTag(id)
        }

    companion object {
        /** Source-only "Detect language" sentinel. */
        val detect = Language("auto", "Detect language", "Detect language")

        /**
         * The languages the model handles well; ordered alphabetically by English name.
         * Verbatim from the iOS `Language.all`.
         */
        val all: List<Language> = listOf(
            Language("ar", "Arabic", "العربية"),
            Language("bn", "Bengali", "বাংলা"),
            Language("bg", "Bulgarian", "Български"),
            Language("zh-Hans", "Chinese (Simplified)", "简体中文"),
            Language("zh-Hant", "Chinese (Traditional)", "繁體中文"),
            Language("cs", "Czech", "Čeština"),
            Language("da", "Danish", "Dansk"),
            Language("nl", "Dutch", "Nederlands"),
            Language("en", "English (US)", "English (US)"),
            Language("fi", "Finnish", "Suomi"),
            Language("fr", "French", "Français"),
            Language("de", "German", "Deutsch"),
            Language("el", "Greek", "Ελληνικά"),
            Language("he", "Hebrew", "עברית"),
            Language("hi", "Hindi", "हिन्दी"),
            Language("hu", "Hungarian", "Magyar"),
            Language("id", "Indonesian", "Indonesia"),
            Language("it", "Italian", "Italiano"),
            Language("ja", "Japanese", "日本語"),
            Language("ko", "Korean", "한국어"),
            Language("ms", "Malay", "Melayu"),
            Language("nb", "Norwegian", "Norsk"),
            Language("fa", "Persian", "فارسی"),
            Language("pl", "Polish", "Polski"),
            Language("pt", "Portuguese", "Português"),
            Language("ro", "Romanian", "Română"),
            Language("ru", "Russian", "Русский"),
            Language("sk", "Slovak", "Slovenčina"),
            Language("es", "Spanish", "Español"),
            Language("sv", "Swedish", "Svenska"),
            Language("ta", "Tamil", "தமிழ்"),
            Language("te", "Telugu", "తెలుగు"),
            Language("th", "Thai", "ไทย"),
            Language("tr", "Turkish", "Türkçe"),
            Language("uk", "Ukrainian", "Українська"),
            Language("vi", "Vietnamese", "Tiếng Việt"),
        )

        /** Source options: "Detect language" first, then every concrete language. */
        val sourceOptions: List<Language> = listOf(detect) + all

        /** Target options: concrete languages only (no "Detect language"). */
        val targetOptions: List<Language> = all

        fun named(id: String): Language = all.firstOrNull { it.id == id } ?: detect
    }
}
