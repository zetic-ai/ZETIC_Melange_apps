//
//  HarmCategory.swift
//  PromptGuard
//

import Foundation

/// Harm categories (S1–S11) for prompt guard classification.
enum HarmCategory: String, CaseIterable, Identifiable {
    case s1 = "S1"
    case s2 = "S2"
    case s3 = "S3"
    case s4 = "S4"
    case s5 = "S5"
    case s6 = "S6"
    case s7 = "S7"
    case s8 = "S8"
    case s9 = "S9"
    case s10 = "S10"
    case s11 = "S11"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .s1: return "Violent Crimes"
        case .s2: return "Non-Violent Crimes"
        case .s3: return "Sex-Related Crimes"
        case .s4: return "Child Sexual Exploitation"
        case .s5: return "Specialized Advice"
        case .s6: return "Privacy"
        case .s7: return "Intellectual Property"
        case .s8: return "Indiscriminate Weapons"
        case .s9: return "Hate"
        case .s10: return "Suicide & Self-Harm"
        case .s11: return "Sexual Content"
        }
    }

    var index: Int {
        HarmCategory.allCases.firstIndex(of: self) ?? 0
    }
}
