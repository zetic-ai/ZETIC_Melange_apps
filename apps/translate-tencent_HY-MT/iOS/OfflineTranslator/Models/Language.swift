import Foundation

/// A translation language. `auto` is a source-only sentinel meaning "let the model
/// infer the source language" — Hunyuan-MT does this naturally because its prompt
/// only needs the *target* language.
struct Language: Identifiable, Hashable {
    let id: String          // stable identifier / rough BCP-47 code
    let englishName: String // name used when building the model prompt
    let nativeName: String  // endonym shown in the picker

    var isDetect: Bool { id == "auto" }

    /// Source-only "Detect language" sentinel.
    static let detect = Language(id: "auto", englishName: "Detect language", nativeName: "Detect language")
}

extension Language {
    /// The languages Tencent HY-MT handles well. Kept broad on purpose so the
    /// live demo can show off coverage; ordered alphabetically by English name.
    static let all: [Language] = [
        Language(id: "ar", englishName: "Arabic", nativeName: "العربية"),
        Language(id: "bn", englishName: "Bengali", nativeName: "বাংলা"),
        Language(id: "bg", englishName: "Bulgarian", nativeName: "Български"),
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
        Language(id: "he", englishName: "Hebrew", nativeName: "עברית"),
        Language(id: "hi", englishName: "Hindi", nativeName: "हिन्दी"),
        Language(id: "hu", englishName: "Hungarian", nativeName: "Magyar"),
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
        Language(id: "sk", englishName: "Slovak", nativeName: "Slovenčina"),
        Language(id: "es", englishName: "Spanish", nativeName: "Español"),
        Language(id: "sv", englishName: "Swedish", nativeName: "Svenska"),
        Language(id: "ta", englishName: "Tamil", nativeName: "தமிழ்"),
        Language(id: "te", englishName: "Telugu", nativeName: "తెలుగు"),
        Language(id: "th", englishName: "Thai", nativeName: "ไทย"),
        Language(id: "tr", englishName: "Turkish", nativeName: "Türkçe"),
        Language(id: "uk", englishName: "Ukrainian", nativeName: "Українська"),
        Language(id: "vi", englishName: "Vietnamese", nativeName: "Tiếng Việt"),
    ]

    /// Source options: "Detect language" first, then every concrete language.
    static let sourceOptions: [Language] = [.detect] + all

    /// Target options: concrete languages only (no "Detect language").
    static let targetOptions: [Language] = all

    static func named(_ id: String) -> Language {
        all.first { $0.id == id } ?? detect
    }

    /// BCP-47-ish code for AVSpeechSynthesizer voice selection.
    var speechCode: String {
        switch id {
        case "auto": return "en-US"
        case "en": return "en-US"
        case "zh-Hans": return "zh-CN"
        case "zh-Hant": return "zh-TW"
        case "nb": return "nb-NO"
        default: return id
        }
    }
}
