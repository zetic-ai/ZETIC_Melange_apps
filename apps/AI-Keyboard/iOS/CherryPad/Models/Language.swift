import Foundation

/// A translation target language. `englishName` is used when building the model
/// prompt; `nativeName` is the endonym shown in the picker. The multilingual Qwen
/// model infers the source language from the text, so only the target is needed.
struct Language: Identifiable, Hashable {
    let id: String          // stable identifier / rough BCP-47 code
    let englishName: String
    let nativeName: String
}

extension Language {
    /// A broad set of languages the small Qwen model handles; ordered alphabetically.
    static let all: [Language] = [
        Language(id: "ar", englishName: "Arabic", nativeName: "العربية"),
        Language(id: "bn", englishName: "Bengali", nativeName: "বাংলা"),
        Language(id: "zh-Hans", englishName: "Chinese (Simplified)", nativeName: "简体中文"),
        Language(id: "zh-Hant", englishName: "Chinese (Traditional)", nativeName: "繁體中文"),
        Language(id: "cs", englishName: "Czech", nativeName: "Čeština"),
        Language(id: "da", englishName: "Danish", nativeName: "Dansk"),
        Language(id: "nl", englishName: "Dutch", nativeName: "Nederlands"),
        Language(id: "en", englishName: "English (US)", nativeName: "English (US)"),
        Language(id: "fi", englishName: "Finnish", nativeName: "Suomi"),
        Language(id: "fr", englishName: "French", nativeName: "Français"),
        Language(id: "de", englishName: "German", nativeName: "Deutsch"),
        Language(id: "el", englishName: "Greek", nativeName: "Ελληνικά"),
        Language(id: "hi", englishName: "Hindi", nativeName: "हिन्दी"),
        Language(id: "id", englishName: "Indonesian", nativeName: "Indonesia"),
        Language(id: "it", englishName: "Italian", nativeName: "Italiano"),
        Language(id: "ja", englishName: "Japanese", nativeName: "日本語"),
        Language(id: "ko", englishName: "Korean", nativeName: "한국어"),
        Language(id: "ms", englishName: "Malay", nativeName: "Melayu"),
        Language(id: "nb", englishName: "Norwegian", nativeName: "Norsk"),
        Language(id: "fa", englishName: "Persian", nativeName: "فارسی"),
        Language(id: "pl", englishName: "Polish", nativeName: "Polski"),
        Language(id: "pt", englishName: "Portuguese", nativeName: "Português"),
        Language(id: "ro", englishName: "Romanian", nativeName: "Română"),
        Language(id: "ru", englishName: "Russian", nativeName: "Русский"),
        Language(id: "es", englishName: "Spanish", nativeName: "Español"),
        Language(id: "sv", englishName: "Swedish", nativeName: "Svenska"),
        Language(id: "th", englishName: "Thai", nativeName: "ไทย"),
        Language(id: "tr", englishName: "Turkish", nativeName: "Türkçe"),
        Language(id: "uk", englishName: "Ukrainian", nativeName: "Українська"),
        Language(id: "vi", englishName: "Vietnamese", nativeName: "Tiếng Việt"),
    ]

    static func named(_ id: String) -> Language {
        all.first { $0.id == id } ?? all.first { $0.id == "en" }!
    }

    /// Lookup used when resolving a handoff payload's `targetLanguage` string.
    static func matching(englishName: String) -> Language? {
        all.first { $0.englishName == englishName }
    }
}
