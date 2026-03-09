import Foundation
import SwiftUI

struct Language: Identifiable, Codable, Hashable {
    let id: String
    let name: String
    let flag: String
    var code: String { id }

    init(id: String, name: String, flag: String = "") {
        self.id = id
        self.name = name
        self.flag = flag
    }

    static func byCode(_ code: String) -> Language {
        catalog.first(where: { $0.id == code }) ?? catalog[0]
    }

    static let catalog: [Language] = [
        Language(id: "en", name: "English", flag: "🇺🇸"),
        Language(id: "es", name: "Spanish", flag: "🇪🇸"),
        Language(id: "fr", name: "French", flag: "🇫🇷"),
        Language(id: "de", name: "German", flag: "🇩🇪"),
        Language(id: "it", name: "Italian", flag: "🇮🇹"),
        Language(id: "pt", name: "Portuguese", flag: "🇵🇹"),
        Language(id: "ru", name: "Russian", flag: "🇷🇺"),
        Language(id: "zh-CN", name: "Chinese (Simplified)", flag: "🇨🇳"),
        Language(id: "zh-TW", name: "Chinese (Traditional)", flag: "🇹🇼"),
        Language(id: "ja", name: "Japanese", flag: "🇯🇵"),
        Language(id: "ko", name: "Korean", flag: "🇰🇷"),
        Language(id: "ar", name: "Arabic", flag: "🇸🇦"),
        Language(id: "hi", name: "Hindi", flag: "🇮🇳"),
        Language(id: "bn", name: "Bengali", flag: "🇧🇩"),
        Language(id: "tr", name: "Turkish", flag: "🇹🇷"),
        Language(id: "vi", name: "Vietnamese", flag: "🇻🇳"),
        Language(id: "th", name: "Thai", flag: "🇹🇭"),
        Language(id: "id", name: "Indonesian", flag: "🇮🇩"),
        Language(id: "nl", name: "Dutch", flag: "🇳🇱"),
        Language(id: "el", name: "Greek", flag: "🇬🇷"),
        Language(id: "he", name: "Hebrew", flag: "🇮🇱"),
        Language(id: "pl", name: "Polish", flag: "🇵🇱"),
        Language(id: "sv", name: "Swedish", flag: "🇸🇪"),
        Language(id: "no", name: "Norwegian", flag: "🇳🇴"),
        Language(id: "da", name: "Danish", flag: "🇩🇰"),
        Language(id: "fi", name: "Finnish", flag: "🇫🇮"),
        Language(id: "cs", name: "Czech", flag: "🇨🇿"),
        Language(id: "sk", name: "Slovak", flag: "🇸🇰"),
        Language(id: "hu", name: "Hungarian", flag: "🇭🇺"),
        Language(id: "ro", name: "Romanian", flag: "🇷🇴"),
        Language(id: "bg", name: "Bulgarian", flag: "🇧🇬"),
        Language(id: "uk", name: "Ukrainian", flag: "🇺🇦"),
        Language(id: "sr", name: "Serbian", flag: "🇷🇸"),
        Language(id: "hr", name: "Croatian", flag: "🇭🇷"),
        Language(id: "sl", name: "Slovenian", flag: "🇸🇮"),
        Language(id: "lt", name: "Lithuanian", flag: "🇱🇹"),
        Language(id: "lv", name: "Latvian", flag: "🇱🇻"),
        Language(id: "et", name: "Estonian", flag: "🇪🇪"),
        Language(id: "fil", name: "Filipino", flag: "🇵🇭"),
        Language(id: "sw", name: "Swahili", flag: "🇰🇪")
    ]
}
