import Foundation

class Tokenizer {
    private var vocab: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]
    
    // Default special tokens
    var bosId = 0
    var eosId = 2
    var unkId = 3
    var padId = 1
    var maskId = 4
    
    init() {
        loadVocab()
    }
    
    private func loadVocab() {
        guard let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
            print("Tokenizer: Failed to load tokenizer.json")
            return
        }
        
        var vocabDict: [String: Any]?
        
        if let model = json["model"] as? [String: Any], let v = model["vocab"] as? [String: Any] {
            vocabDict = v
        } else if let v = json["vocab"] as? [String: Any] {
            vocabDict = v
        }
        
        if let vocabDict = vocabDict {
            for (token, idAny) in vocabDict {
                if let id = idAny as? Int {
                    vocab[token] = id
                    idToToken[id] = token
                }
            }
        }
        
        // Update special tokens if possible
        bosId = vocab["<s>"] ?? vocab["[CLS]"] ?? bosId
        eosId = vocab["</s>"] ?? vocab["[SEP]"] ?? eosId
        unkId = vocab["<unk>"] ?? vocab["[UNK]"] ?? unkId
        padId = vocab["<pad>"] ?? vocab["[PAD]"] ?? padId
        maskId = vocab["<mask>"] ?? vocab["[MASK]"] ?? maskId
        
        print("Tokenizer: Loaded \(vocab.count) tokens")
    }
    
    func encode(_ text: String) -> [Int] {
        var ids: [Int] = [bosId]
        
        // Simplified BPE-like Greedy Match
        // 1. Prepend space
        // 2. Replace space with Ġ (\u0120)
        let processedText = " " + text
        let cleanText = processedText.replacingOccurrences(of: " ", with: "\u{0120}")
        
        var i = 0
        let chars = Array(cleanText)
        let len = chars.count
        
        while i < len {
            var matchFound = false
            let maxSearchLen = min(len - i, 20)
            
            for l in stride(from: maxSearchLen, through: 1, by: -1) {
                let sub = String(chars[i..<i+l])
                if let id = vocab[sub] {
                    ids.append(id)
                    i += l
                    matchFound = true
                    break
                }
            }
            
            if !matchFound {
                ids.append(unkId)
                i += 1
            }
        }
        
        ids.append(eosId)
        return ids
    }
    
    func decode(_ tokens: [Int]) -> String {
        var result = ""
        for t in tokens {
            if t == bosId || t == eosId || t == padId { continue }
            if let s = idToToken[t] {
                // Remove Ġ and append
                result += s.replacingOccurrences(of: "\u{0120}", with: " ")
            }
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    func decodeToken(_ id: Int) -> String {
        if let s = idToToken[id] {
            return s.replacingOccurrences(of: "\u{0120}", with: " ")
        }
        return ""
    }
    
    func getRawToken(_ id: Int) -> String? {
        return idToToken[id]
    }
}
