package ai.zetic.demo.cherrypad.model

/**
 * A translation target language. [englishName] is used when building the model prompt;
 * [nativeName] is the endonym shown in the picker. The multilingual model infers the
 * source language from the text, so only the target is needed.
 */
data class Language(
    val id: String,          // stable identifier / rough BCP-47 code
    val englishName: String,
    val nativeName: String,
) {
    companion object {
        /** A broad set of languages the small model handles; ordered alphabetically. */
        val all: List<Language> = listOf(
            Language("ar", "Arabic", "العربية"),
            Language("bn", "Bengali", "বাংলা"),
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
            Language("hi", "Hindi", "हिन्दी"),
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
            Language("es", "Spanish", "Español"),
            Language("sv", "Swedish", "Svenska"),
            Language("th", "Thai", "ไทย"),
            Language("tr", "Turkish", "Türkçe"),
            Language("uk", "Ukrainian", "Українська"),
            Language("vi", "Vietnamese", "Tiếng Việt"),
        )

        /** Default translation target. */
        val default: Language get() = named("ko")

        fun named(id: String): Language =
            all.firstOrNull { it.id == id } ?: all.first { it.id == "en" }

        /** Lookup used when resolving a saved `targetLanguage` string. */
        fun matching(englishName: String): Language? =
            all.firstOrNull { it.englishName == englishName }
    }
}
