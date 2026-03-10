//
//  Tokenizer.swift
//  PromptGuard
//
//  Local tokenizer matching meta-llama/Llama-Prompt-Guard-2-86M (or same vocab as the CoreML model).
//  Loads from bundle: tokenizer (3).json / tokenizer.json (and optionally tokenizer_config / special_tokens_map).
//  Same pattern as TextAnonymizer: no cloud or HF calls.
//

import Foundation

final class PromptGuardTokenizer {
    private var vocab: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]

    var bosId: Int = 1
    var eosId: Int = 2
    var unkId: Int = 0
    var padId: Int = 0

    /// Bundle resource names to try (your files: "tokenizer (3)", then standard "tokenizer").
    private static let tokenizerResourceNames = ["tokenizer (3)", "tokenizer"]

    private let loadLock = NSLock()

    init() {
        // Don't load here – load on first use (ensureLoaded) so we don't block main thread with large JSON.
    }

    /// Call before encode/createInput; loads vocab from bundle on first call (safe to call from background).
    func ensureLoaded() {
        loadLock.lock()
        defer { loadLock.unlock() }
        if vocab.isEmpty {
            loadVocab()
        }
    }

    private func loadVocab() {
        var json: [String: Any]?
        var loadedFrom: String?
        for name in Self.tokenizerResourceNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "json", subdirectory: nil),
               let data = try? Data(contentsOf: url),
               let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                json = parsed
                loadedFrom = name
                break
            }
        }
        guard let json = json else {
            print("PromptGuard Tokenizer: No tokenizer.json found. Tried: \(Self.tokenizerResourceNames.joined(separator: ", ")).")
            return
        }

        // 1) model.vocab: can be dict (token -> id) or Unigram array [[token, score], ...] where index = id
        if let model = json["model"] as? [String: Any] {
            if let vocabDict = model["vocab"] as? [String: Any] {
                for (token, idAny) in vocabDict {
                    if let id = idAny as? Int {
                        vocab[token] = id
                        idToToken[id] = token
                    }
                }
            } else if let vocabArray = model["vocab"] as? [[Any]] {
                for (id, entry) in vocabArray.enumerated() {
                    if let token = entry.first as? String {
                        vocab[token] = id
                        idToToken[id] = token
                    }
                }
            }
        } else if let vocabDict = json["vocab"] as? [String: Any] {
            for (token, idAny) in vocabDict {
                if let id = idAny as? Int {
                    vocab[token] = id
                    idToToken[id] = token
                }
            }
        }

        // 2) added_tokens: [{ "id": n, "content": "..." }, ...]
        if let added = json["added_tokens"] as? [[String: Any]] {
            for item in added {
                if let content = item["content"] as? String, let id = item["id"] as? Int {
                    vocab[content] = id
                    idToToken[id] = content
                }
            }
        }

        // 3) Special token IDs (this tokenizer uses [PAD]=0, [CLS]=1, [SEP]=2, [UNK]=3)
        bosId = vocab["<s>"] ?? vocab["<|begin_of_text|>"] ?? vocab["[CLS]"] ?? 1
        eosId = vocab["</s>"] ?? vocab["<|end_of_text|>"] ?? vocab["[SEP]"] ?? 2
        unkId = vocab["<unk>"] ?? vocab["<|unk|>"] ?? vocab["[UNK]"] ?? 0
        padId = vocab["<pad>"] ?? vocab["<|pad|>"] ?? vocab["[PAD]"] ?? 0

        if let from = loadedFrom {
            print("PromptGuard Tokenizer: Loaded \(vocab.count) tokens from '\(from).json' (pad=\(padId), bos=\(bosId), eos=\(eosId))")
        } else {
            print("PromptGuard Tokenizer: Loaded \(vocab.count) tokens (pad=\(padId), bos=\(bosId), eos=\(eosId))")
        }
    }

    var isLoaded: Bool { !vocab.isEmpty }

    /// Encode text to token IDs (same tokenizer as Python: tokenizer(text, return_tensors="pt")).
    /// Greedy longest-match from vocab; supports ▁ (SentencePiece) and Ġ (RoBERTa) space style.
    func encode(_ text: String) -> [Int] {
        var ids: [Int] = [bosId]
        let withLeadingSpace = " " + text
        let hasSentencePiece = vocab.keys.contains { $0.hasPrefix("\u{2581}") }
        let processed: String
        let spaceSub: String
        if hasSentencePiece {
            processed = withLeadingSpace.replacingOccurrences(of: " ", with: "\u{2581}")
            spaceSub = "\u{2581}"
        } else {
            processed = withLeadingSpace.replacingOccurrences(of: " ", with: "\u{0120}")
            spaceSub = "\u{0120}"
        }
        ids.append(contentsOf: greedyEncode(Array(processed), spaceSub: spaceSub))
        ids.append(eosId)
        return ids
    }

    private func greedyEncode(_ chars: [Character], spaceSub: String) -> [Int] {
        var ids: [Int] = []
        let len = chars.count
        var i = 0
        while i < len {
            var found = false
            for l in stride(from: min(32, len - i), through: 1, by: -1) {
                let sub = String(chars[i ..< i + l])
                if let id = vocab[sub] {
                    ids.append(id)
                    i += l
                    found = true
                    break
                }
            }
            if !found {
                let single = String(chars[i])
                ids.append(vocab[spaceSub + single] ?? vocab[single] ?? unkId)
                i += 1
            }
        }
        return ids
    }
}
