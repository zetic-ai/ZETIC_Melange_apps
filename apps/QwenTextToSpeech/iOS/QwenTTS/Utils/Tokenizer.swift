import Foundation

class Tokenizer {
    private var encoder: [String: Int] = [:]
    private var decoder: [Int: String] = [:]
    private var bpeRanks: [String: Int] = [:]
    private var cache: [String: String] = [:]
    
    // Special tokens mapping
    private let specialTokens: [String: Int] = [
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|audio_start|>": 151669,
        "<|audio_end|>": 151670,
        "<tts_pad>": 151671,
        "<tts_text_bos>": 151672,
        "<tts_text_eod>": 151673
    ]
    
    init() {
        loadVocab()
        loadMerges()
    }
    
    private func loadVocab() {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Int] else {
            print("Failed to load vocab.json")
            return
        }
        self.encoder = json
        self.decoder = Dictionary(uniqueKeysWithValues: json.map { ($1, $0) })
    }
    
    private func loadMerges() {
        guard let url = Bundle.main.url(forResource: "merges", withExtension: "txt"),
              let content = try? String(contentsOf: url, encoding: .utf8) else {
            print("Failed to load merges.txt")
            return
        }
        
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        for (i, line) in lines.enumerated() {
            self.bpeRanks[line] = i
        }
    }
    
    func encode(_ text: String) -> [Int] {
        // Robust GPT-2 style pre-tokenization using Regex
        // Pattern matches:
        // 1. Contractions ('s, 't, 're, 've, 'm, 'll, 'd)
        // 2. Letters with optional leading space ( ?\p{L}+)
        // 3. Numbers with optional leading space ( ?\p{N}+)
        // 4. Anything else (punctuation/symbols) excluding space ( ?[^\s\p{L}\p{N}]+)
        // 5. Trailing whitespace (\s+(?!\S))
        // 6. Remaining whitespace (\s+)
        
        let pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/
        
        // Find all matches
        let matches = text.matches(of: pattern)
        let words = matches.map { String($0.output) }
        
        var tokens: [Int] = []
        
        for word in words {
            let bpeTokens = bpe(token: word)
            tokens.append(contentsOf: bpeTokens.compactMap { encoder[$0] })
        }
        
        return tokens
    }
    
    private func bpe(token: String) -> [String] {
        if let cached = cache[token] {
            return cached.components(separatedBy: " ")
        }
        
        var word = token.map { String($0) }
        var pairs = getPairs(word: word)
        
        if pairs.isEmpty {
            return [token]
        }
        
        while true {
            var bigram: (String, String)? = nil
            var minRank = Int.max
            
            for pair in pairs {
                let key = pair.0 + " " + pair.1
                if let rank = bpeRanks[key], rank < minRank {
                    minRank = rank
                    bigram = pair
                }
            }
            
            guard let best = bigram else { break }
            
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == best.0 && word[i+1] == best.1 {
                    newWord.append(best.0 + best.1)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            
            word = newWord
            if word.count == 1 { break }
            pairs = getPairs(word: word)
        }
        
        let result = word
        cache[token] = result.joined(separator: " ")
        return result
    }
    
    private func getPairs(word: [String]) -> [(String, String)] {
        var pairs: [(String, String)] = []
        guard word.count > 1 else { return pairs }
        for i in 0..<word.count-1 {
            pairs.append((word[i], word[i+1]))
        }
        return pairs
    }
    
    func getSpecialTokenID(_ token: String) -> Int? {
        return specialTokens[token]
    }
}
