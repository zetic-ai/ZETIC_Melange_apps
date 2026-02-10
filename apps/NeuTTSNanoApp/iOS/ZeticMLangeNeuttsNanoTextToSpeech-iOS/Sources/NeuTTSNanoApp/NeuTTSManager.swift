//
//  NeuTTSManager.swift
//  NeuTTSNanoApp
//
//  Created by Assistant
//

import Foundation
import AVFoundation
import ZeticMLange

class NeuTTSManager: ObservableObject {
    private let tokenKey = "dev_24c61ecfff874298a52c5f3be0a83f71"

    // Model configurations
    private let backboneModelName = "jathin-zetic/neutts_nano"
    private let backboneVersion = 1

    private let encoderModelName = "jathin-zetic/neucodec-encoder"
    private let encoderVersion = 1

    private let decoderModelName = "jathin-zetic/neucodec-decoder"
    private let decoderVersion = 2

    // Models
    private var backboneModel: ZeticMLangeModel?
    private var encoderModel: ZeticMLangeModel?
    private var decoderModel: ZeticMLangeModel?
    private var tokenizerVocab: [String: Int32]?
    private var tokenizerIdToToken: [Int32: String]?
    private var tokenizerSpecialTokens: Set<String> = []
    private var byteEncoder: [UInt8: String]?
    private var tokenizerRegex: String?
    private var tokenizerAddPrefixSpace: Bool = false
    private var tokenizerBosTokenId: Int32?
    private var tokenizerPadTokenId: Int32?

    @Published var isInitialized = false
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var statusMessage: String = "Idle"
    @Published var logLines: [String] = []

    // Audio player
    private var audioPlayer: AVAudioPlayer?

    init() {
        setupAudioSession()
        Task {
            await initializeModels()
        }
    }

    private func setupAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            appendLog("Audio session setup failed: \(error.localizedDescription)")
        }
    }

    private func initializeModels() async {
        do {
            await MainActor.run {
                statusMessage = "Initializing models..."
                errorMessage = nil
            }

            // Load only the core models at startup to reduce initial wait.
            appendLog("Loading backbone model: \(backboneModelName) v\(backboneVersion)")
            backboneModel = try ZeticMLangeModel(
                tokenKey: tokenKey,
                name: backboneModelName,
                version: backboneVersion,
                onDownload: { [weak self] progress in
                    Task { @MainActor in
                        self?.statusMessage = String(format: "Downloading backbone model: %.0f%%", progress * 100)
                        self?.appendLog(String(format: "Backbone download progress: %.1f%%", progress * 100))
                    }
                }
            )
            appendLog("Backbone model loaded")

            appendLog("Loading decoder model: \(decoderModelName) v\(decoderVersion)")
            decoderModel = try ZeticMLangeModel(
                tokenKey: tokenKey,
                name: decoderModelName,
                version: decoderVersion,
                onDownload: { [weak self] progress in
                    Task { @MainActor in
                        self?.statusMessage = String(format: "Downloading decoder model: %.0f%%", progress * 100)
                        self?.appendLog(String(format: "Decoder download progress: %.1f%%", progress * 100))
                    }
                }
            )
            appendLog("Decoder model loaded")

            await MainActor.run {
                isInitialized = true
                errorMessage = nil
                statusMessage = "Models ready (encoder loads on demand)"
            }

        } catch {
            await MainActor.run {
                errorMessage = "Failed to initialize models: \(error.localizedDescription)"
                isInitialized = false
                statusMessage = "Initialization failed"
            }
            appendLog("Model initialization failed: \(error.localizedDescription)")
        }
    }

    private func loadEncoderModelIfNeeded() async throws {
        if encoderModel != nil {
            return
        }
        await MainActor.run {
            statusMessage = "Loading encoder model..."
        }
        appendLog("Loading encoder model: \(encoderModelName) v\(encoderVersion)")
        encoderModel = try ZeticMLangeModel(
            tokenKey: tokenKey,
            name: encoderModelName,
            version: encoderVersion,
            onDownload: { [weak self] progress in
                Task { @MainActor in
                    self?.statusMessage = String(format: "Downloading encoder model: %.0f%%", progress * 100)
                    self?.appendLog(String(format: "Encoder download progress: %.1f%%", progress * 100))
                }
            }
        )
        appendLog("Encoder model loaded")
        await MainActor.run {
            statusMessage = "Models ready"
        }
    }

    func synthesizeSpeech(text: String, referenceAudioData: Data?, referenceText: String?) async throws -> Data {
        guard let backboneModel = backboneModel,
              let decoderModel = decoderModel else {
            throw NSError(domain: "NeuTTS", code: -1, userInfo: [NSLocalizedDescriptionKey: "Models not initialized"])
        }

        await MainActor.run {
            isProcessing = true
            errorMessage = nil
        }

        defer {
            Task { @MainActor in
                isProcessing = false
            }
        }

        logBundleResourceStatus()
        let refCodes = try await encodeReferenceAudio(referenceAudioData)
        appendLog("Reference codes count: \(refCodes.count)")

        // Tokenize input text using Hugging Face-compatible tokenizer config
        let promptIds = try buildPromptIds(refCodes: refCodes, refText: referenceText ?? "", inputText: text)
        appendLog("Prompt token count: \(promptIds.count)")
        let attentionMask = makeAttentionMask(for: promptIds, maxLength: 128)

        // Prepare inputs for backbone model
        // This is a simplified version - you'd need to match the exact tensor shapes
        // from the model's requirements
        let backboneInputs = prepareBackboneInputs(
            inputTokens: promptIds,
            attentionMask: attentionMask
        )

        // Run backbone model
        let backboneOutputs = try backboneModel.run(inputs: backboneInputs)

        // Extract codes from backbone output
        let codes = try extractCodesFromBackboneOutput(backboneOutputs, promptLength: min(promptIds.count, 128))
        appendLog("Decoded speech token count: \(codes.count)")

        // Decode audio using decoder model
        let decoderInputs = prepareDecoderInputs(codes: codes)
        let decoderOutputs = try decoderModel.run(inputs: decoderInputs)

        // Convert output to audio data
        let audioData = try convertOutputToAudioData(decoderOutputs)

        return audioData
    }

    private func encodeReferenceAudio(_ audioData: Data?) async throws -> [Int32] {
        // If no reference audio provided, return default codes
        guard let audioData = audioData else {
            // Return some default reference codes (this should be replaced with actual encoding)
            return [Int32](repeating: 0, count: 50)
        }

        try await loadEncoderModelIfNeeded()
        guard let encoderModel = encoderModel else {
            throw NSError(domain: "NeuTTS", code: -1, userInfo: [NSLocalizedDescriptionKey: "Encoder model not available"])
        }

        // Convert audio data to model input format
        // This is simplified - you'd need to properly convert audio to the expected tensor format
        let encoderInputs = try prepareEncoderInputs(audioData: audioData)
        let encoderOutputs = try encoderModel.run(inputs: encoderInputs)

        // Extract reference codes from encoder output
        return extractCodesFromEncoderOutput(encoderOutputs)
    }

    private func tokenizeText(_ text: String) throws -> [Int32] {
        guard let vocab = loadTokenizerIfNeeded(),
              let regex = tokenizerRegex,
              let bosTokenId = tokenizerBosTokenId,
              let padTokenId = tokenizerPadTokenId else {
            throw NSError(domain: "Tokenizer", code: -1, userInfo: [NSLocalizedDescriptionKey: "Tokenizer.json not available. Add tokenizer (2).json to the app bundle or Documents."])
        }

        let tokens = regexPreTokenize(text: text, pattern: regex, addPrefixSpace: tokenizerAddPrefixSpace)

        var ids: [Int32] = [bosTokenId]
        let encoder = byteEncoder ?? buildByteEncoder()
        byteEncoder = encoder

        for token in tokens {
            let byteLevel = byteLevelEncode(token, encoder: encoder)
            for ch in byteLevel {
                let key = String(ch)
                if let id = vocab[key] {
                    ids.append(id)
                } else {
                    ids.append(padTokenId)
                }
            }
        }

        return ids
    }

    private func prepareBackboneInputs(inputTokens: [Int32], attentionMask: [Int32]) -> [Tensor] {
        // Create tensors for backbone model inputs
        // This is a placeholder - you need to match the exact input requirements of the model

        // Convert tokens to tensor data
        let inputTensorData = padOrTruncate(tokens: inputTokens, maxLength: 128)

        // Create tensors (shapes need to match model expectations)
        let inputTensor = Tensor(
            data: int32ArrayToData(inputTensorData),
            dataType: BuiltinDataType.int32,
            shape: [1, 128]
        )
        let attentionMaskTensor = Tensor(
            data: int32ArrayToData(attentionMask),
            dataType: BuiltinDataType.int32,
            shape: [1, 128]
        )
        return [inputTensor, attentionMaskTensor]
    }

    private func prepareEncoderInputs(audioData: Data) throws -> [Tensor] {
        // Convert audio data to tensor format expected by encoder
        // Expected input: audio shape [1, 1, 16000], float32
        let samples = try decodeWavToMonoFloat32(audioData: audioData, targetSampleRate: 16000)
        let fixedSamples = trimOrPad(samples: samples, length: 16000)

        let audioTensor = Tensor(
            data: floatArrayToData(fixedSamples),
            dataType: BuiltinDataType.float32,
            shape: [1, 1, 16000]
        )
        return [audioTensor]
    }

    private func prepareDecoderInputs(codes: [Int32]) -> [Tensor] {
        // Prepare inputs for decoder model
        let fixedCodes = trimOrPad(codes: codes, length: 50)
        let int64Codes = fixedCodes.map { Int64($0) }
        let codesTensor = Tensor(
            data: int64ArrayToData(int64Codes),
            dataType: BuiltinDataType.int64,
            shape: [1, 1, 50]
        )
        return [codesTensor]
    }

    private func extractCodesFromBackboneOutput(_ outputs: [Tensor], promptLength: Int) throws -> [Int32] {
        // Extract the generated speech codes from the LM output.
        guard let outputTensor = outputs.first else {
            return []
        }

        let outputIds = dataToInt32Array(outputTensor.data)
        if outputIds.isEmpty {
            throw NSError(domain: "NeuTTS", code: -3, userInfo: [NSLocalizedDescriptionKey: "Backbone output is empty"])
        }

        // Drop the prompt portion if the model returned the full sequence.
        let generatedIds = outputIds.count > promptLength ? Array(outputIds.dropFirst(promptLength)) : outputIds
        let outputText = decodeTokensToString(ids: generatedIds)
        appendLog("Backbone output preview: \(String(outputText.prefix(200)))")
        let speechIds = extractSpeechIds(from: outputText)

        if speechIds.isEmpty {
            throw NSError(domain: "NeuTTS", code: -4, userInfo: [NSLocalizedDescriptionKey: "No speech tokens found in backbone output"])
        }

        return speechIds
    }

    private func extractCodesFromEncoderOutput(_ outputs: [Tensor]) -> [Int32] {
        // Extract reference codes from encoder output
        guard let outputTensor = outputs.first else {
            return []
        }

        return dataToInt32Array(outputTensor.data)
    }

    private func convertOutputToAudioData(_ outputs: [Tensor]) throws -> Data {
        // Convert decoder output to audio data
        guard let outputTensor = outputs.first else {
            throw NSError(domain: "NeuTTS", code: -1, userInfo: [NSLocalizedDescriptionKey: "No output from decoder"])
        }

        // Convert float samples back to 16-bit PCM
        var pcmData = Data()
        let samples = DataUtils.dataToFloatArray(outputTensor.data)

        for sample in samples {
            // Convert from [-1, 1] to Int16
            let clampedSample = max(-1.0, min(1.0, sample))
            let int16Sample = Int16(clampedSample * 32767.0)
            pcmData.append(contentsOf: withUnsafeBytes(of: int16Sample) { Data($0) })
        }

        return makeWavData(pcmData: pcmData, sampleRate: 24000, channels: 1, bitsPerSample: 16)
    }

    private func floatArrayToData(_ values: [Float]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func int32ArrayToData(_ values: [Int32]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func int64ArrayToData(_ values: [Int64]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func makeWavData(pcmData: Data, sampleRate: Int, channels: Int, bitsPerSample: Int) -> Data {
        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8
        let dataSize = pcmData.count
        let riffChunkSize = 36 + dataSize

        var header = Data()
        header.append(contentsOf: "RIFF".utf8)
        header.append(contentsOf: withUnsafeBytes(of: UInt32(riffChunkSize).littleEndian, { Data($0) }))
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        header.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian, { Data($0) })) // PCM header size
        header.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian, { Data($0) })) // PCM format
        header.append(contentsOf: withUnsafeBytes(of: UInt16(channels).littleEndian, { Data($0) }))
        header.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian, { Data($0) }))
        header.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian, { Data($0) }))
        header.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian, { Data($0) }))
        header.append(contentsOf: withUnsafeBytes(of: UInt16(bitsPerSample).littleEndian, { Data($0) }))
        header.append(contentsOf: "data".utf8)
        header.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian, { Data($0) }))

        var wav = Data()
        wav.append(header)
        wav.append(pcmData)
        return wav
    }

    private func dataToInt32Array(_ data: Data) -> [Int32] {
        data.withUnsafeBytes { rawBuffer -> [Int32] in
            let buffer = rawBuffer.bindMemory(to: Int32.self)
            return Array(buffer)
        }
    }

    private func trimOrPad(samples: [Float], length: Int) -> [Float] {
        if samples.count == length { return samples }
        if samples.count > length { return Array(samples.prefix(length)) }
        return samples + Array(repeating: 0, count: length - samples.count)
    }

    private func trimOrPad(codes: [Int32], length: Int) -> [Int32] {
        if codes.count == length { return codes }
        if codes.count > length { return Array(codes.prefix(length)) }
        return codes + Array(repeating: 0, count: length - codes.count)
    }

    private func padOrTruncate(tokens: [Int32], maxLength: Int) -> [Int32] {
        if tokens.count == maxLength { return tokens }
        if tokens.count > maxLength { return Array(tokens.prefix(maxLength)) }
        return tokens + Array(repeating: 0, count: maxLength - tokens.count)
    }

    private func makeAttentionMask(for tokens: [Int32], maxLength: Int) -> [Int32] {
        let clampedLength = min(tokens.count, maxLength)
        let ones = Array(repeating: Int32(1), count: clampedLength)
        let zeros = Array(repeating: Int32(0), count: maxLength - clampedLength)
        return ones + zeros
    }

    private func buildPromptIds(refCodes: [Int32], refText: String, inputText: String) throws -> [Int32] {
        guard let vocab = loadTokenizerIfNeeded() else {
            throw NSError(domain: "Tokenizer", code: -2, userInfo: [NSLocalizedDescriptionKey: "Tokenizer not loaded"])
        }

        let refTextPhones = phonemize(refText)
        let inputTextPhones = phonemize(inputText)
        let combinedText = refTextPhones.isEmpty ? inputTextPhones : "\(refTextPhones) \(inputTextPhones)"

        let speechReplace = "<|SPEECH_REPLACE|>"
        let speechGenStart = "<|SPEECH_GENERATION_START|>"
        let textReplace = "<|TEXT_REPLACE|>"
        let textPromptStart = "<|TEXT_PROMPT_START|>"
        let textPromptEnd = "<|TEXT_PROMPT_END|>"

        guard let speechReplaceId = vocab[speechReplace],
              let speechGenStartId = vocab[speechGenStart],
              let textReplaceId = vocab[textReplace],
              let textPromptStartId = vocab[textPromptStart],
              let textPromptEndId = vocab[textPromptEnd] else {
            throw NSError(domain: "Tokenizer", code: -3, userInfo: [NSLocalizedDescriptionKey: "Missing special token IDs in tokenizer vocab"])
        }

        let chat = "user: Convert the text to speech:\(textReplace)\nassistant:\(speechReplace)"
        var ids = try encodeWithSpecialTokens(chat)

        guard let textReplaceIdx = ids.firstIndex(of: textReplaceId),
              let speechReplaceIdx = ids.firstIndex(of: speechReplaceId) else {
            throw NSError(domain: "Tokenizer", code: -4, userInfo: [NSLocalizedDescriptionKey: "Failed to locate placeholder tokens in template"])
        }

        let inputIds = try encodePlainText(combinedText)
        ids = ids.prefix(upTo: textReplaceIdx)
            + [textPromptStartId]
            + inputIds
            + [textPromptEndId]
            + ids.suffix(from: textReplaceIdx + 1)

        let codesStr = refCodes.map { "<|speech_\($0)|>" }.joined()
        let codesIds = try encodeWithSpecialTokens(codesStr)
        let prefix = ids.prefix(upTo: speechReplaceIdx)
        ids = Array(prefix) + [speechGenStartId] + codesIds

        return ids
    }

    private func encodePlainText(_ text: String) throws -> [Int32] {
        guard let vocab = loadTokenizerIfNeeded(),
              let regex = tokenizerRegex,
              let padTokenId = tokenizerPadTokenId else {
            throw NSError(domain: "Tokenizer", code: -5, userInfo: [NSLocalizedDescriptionKey: "Tokenizer not initialized"])
        }

        let tokens = regexPreTokenize(text: text, pattern: regex, addPrefixSpace: tokenizerAddPrefixSpace)
        let encoder = byteEncoder ?? buildByteEncoder()
        byteEncoder = encoder

        var ids: [Int32] = []
        for token in tokens {
            let byteLevel = byteLevelEncode(token, encoder: encoder)
            for ch in byteLevel {
                let key = String(ch)
                if let id = vocab[key] {
                    ids.append(id)
                } else {
                    ids.append(padTokenId)
                }
            }
        }
        return ids
    }

    private func encodeWithSpecialTokens(_ text: String) throws -> [Int32] {
        guard let vocab = loadTokenizerIfNeeded(),
              let padTokenId = tokenizerPadTokenId else {
            throw NSError(domain: "Tokenizer", code: -6, userInfo: [NSLocalizedDescriptionKey: "Tokenizer not initialized"])
        }

        let specials = tokenizerSpecialTokens
        if specials.isEmpty {
            return try encodePlainText(text)
        }

        var ids: [Int32] = []
        var remaining = text[...]
        while !remaining.isEmpty {
            if let match = specials.first(where: { remaining.hasPrefix($0) }) {
                if let id = vocab[match] {
                    ids.append(id)
                } else {
                    ids.append(padTokenId)
                }
                remaining = remaining.dropFirst(match.count)
                continue
            }

            let nextSpecialIndex = specials.compactMap { remaining.range(of: $0)?.lowerBound }.min()
            let chunkEnd = nextSpecialIndex ?? remaining.endIndex
            let chunk = String(remaining[..<chunkEnd])
            ids.append(contentsOf: try encodePlainText(chunk))
            remaining = remaining[chunkEnd...]
        }

        return ids
    }

    private func decodeTokensToString(ids: [Int32]) -> String {
        guard let reverse = tokenizerIdToToken else { return "" }
        return ids.compactMap { reverse[$0] }.joined()
    }

    private func extractSpeechIds(from text: String) -> [Int32] {
        let pattern = "<\\|speech_(\\d+)\\|>"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return []
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, options: [], range: range).compactMap { match in
            guard let range = Range(match.range(at: 1), in: text) else { return nil }
            return Int32(text[range]) ?? nil
        }
    }

    private func phonemize(_ text: String) -> String {
        let phones = EspeakPhonemizer.shared.phonemize(text).trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        appendLog("Phonemizer sample: \(String(phones.prefix(120)))")
        return phones
    }

    private func logBundleResourceStatus() {
        let tokenizerURL = Bundle.main.url(forResource: "tokenizer (2)", withExtension: "json")
            ?? Bundle.main.url(forResource: "tokenizer", withExtension: "json")
        let tokensURL = Bundle.main.url(forResource: "special_tokens_map (1)", withExtension: "json")
            ?? Bundle.main.url(forResource: "special_tokens_map", withExtension: "json")
        let dataURL = Bundle.main.url(forResource: "espeak-ng-data", withExtension: nil)
        appendLog("Bundle tokenizer.json: \(tokenizerURL?.path ?? "missing")")
        appendLog("Bundle special_tokens_map.json: \(tokensURL?.path ?? "missing")")
        appendLog("Bundle espeak-ng-data: \(dataURL?.path ?? "missing")")
    }

    private func loadTokenizerIfNeeded() -> [String: Int32]? {
        if let vocab = tokenizerVocab {
            return vocab
        }

        let tokenizerFileNames = ["tokenizer.json", "tokenizer (2).json"]
        let fileManager = FileManager.default
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first

        let urlToUse: URL? = {
            if let documentsURL {
                for name in tokenizerFileNames {
                    let docURL = documentsURL.appendingPathComponent(name)
                    if fileManager.fileExists(atPath: docURL.path) { return docURL }
                }
            }
            for name in tokenizerFileNames {
                if let bundleURL = Bundle.main.url(forResource: name.replacingOccurrences(of: ".json", with: ""), withExtension: "json") {
                    return bundleURL
                }
            }
            return nil
        }()

        guard let url = urlToUse else {
            appendLog("Missing tokenizer.json. Using fallback tokenizer.")
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
            guard let model = json?["model"] as? [String: Any],
                  let vocab = model["vocab"] as? [String: Any] else {
                appendLog("Tokenizer vocab not found in tokenizer.json.")
                return nil
            }

            if let preTokenizer = json?["pre_tokenizer"] as? [String: Any],
               let preTokenizers = preTokenizer["pretokenizers"] as? [[String: Any]] {
                for item in preTokenizers {
                    if let type = item["type"] as? String, type == "Split",
                       let pattern = item["pattern"] as? [String: Any],
                       let regex = pattern["Regex"] as? String {
                        tokenizerRegex = regex
                    }
                    if let type = item["type"] as? String, type == "ByteLevel",
                       let addPrefix = item["add_prefix_space"] as? Bool {
                        tokenizerAddPrefixSpace = addPrefix
                    }
                }
            }

            var mapped: [String: Int32] = [:]
            mapped.reserveCapacity(vocab.count)
            for (key, value) in vocab {
                if let intVal = value as? Int {
                    mapped[key] = Int32(intVal)
                } else if let intVal = value as? Int32 {
                    mapped[key] = intVal
                } else if let strVal = value as? String, let intVal = Int32(strVal) {
                    mapped[key] = intVal
                }
            }

            if let addedTokens = json?["added_tokens"] as? [[String: Any]] {
                for token in addedTokens {
                    if let content = token["content"] as? String,
                       let idValue = token["id"] {
                        if let intVal = idValue as? Int {
                            mapped[content] = Int32(intVal)
                        } else if let intVal = idValue as? Int32 {
                            mapped[content] = intVal
                        } else if let strVal = idValue as? String, let intVal = Int32(strVal) {
                            mapped[content] = intVal
                        }
                    }
                }
            }

            let (bosToken, padToken) = loadSpecialTokens()
            tokenizerBosTokenId = mapped[bosToken] ?? 128000
            tokenizerPadTokenId = mapped[padToken] ?? 128001

            tokenizerVocab = mapped
            tokenizerIdToToken = Dictionary(uniqueKeysWithValues: mapped.map { ($0.value, $0.key) })
            tokenizerSpecialTokens = Set(mapped.keys.filter { $0.hasPrefix("<|") && $0.hasSuffix("|>") })
            appendLog("Loaded tokenizer vocab with \(mapped.count) entries")
            return mapped
        } catch {
            appendLog("Failed to load tokenizer.json: \(error.localizedDescription)")
            return nil
        }
    }

    private func regexPreTokenize(text: String, pattern: String, addPrefixSpace: Bool) -> [String] {
        let input = addPrefixSpace ? " " + text : text
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return [input]
        }
        let range = NSRange(input.startIndex..<input.endIndex, in: input)
        return regex.matches(in: input, options: [], range: range).compactMap { match in
            guard let range = Range(match.range, in: input) else { return nil }
            return String(input[range])
        }
    }

    private func loadSpecialTokens() -> (bos: String, pad: String) {
        let fileManager = FileManager.default
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        let fileNames = ["special_tokens_map.json", "special_tokens_map (1).json"]

        let urlToUse: URL? = {
            if let documentsURL {
                for name in fileNames {
                    let docURL = documentsURL.appendingPathComponent(name)
                    if fileManager.fileExists(atPath: docURL.path) { return docURL }
                }
            }
            for name in fileNames {
                if let bundleURL = Bundle.main.url(forResource: name.replacingOccurrences(of: ".json", with: ""), withExtension: "json") {
                    return bundleURL
                }
            }
            return nil
        }()

        guard let url = urlToUse,
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
              let bos = (json["bos_token"] as? [String: Any])?["content"] as? String,
              let pad = (json["pad_token"] as? [String: Any])?["content"] as? String else {
            return ("<|begin_of_text|>", "<|end_of_text|>")
        }
        return (bos, pad)
    }

    private func buildByteEncoder() -> [UInt8: String] {
        var bs = [UInt8]()
        bs.append(contentsOf: UInt8(33)...UInt8(126))
        bs.append(contentsOf: UInt8(161)...UInt8(172))
        bs.append(contentsOf: UInt8(174)...UInt8(255))

        var cs = bs.map { Int($0) }
        var n = 0
        for b in UInt8(0)...UInt8(255) {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }

        var encoder: [UInt8: String] = [:]
        encoder.reserveCapacity(bs.count)
        for (b, c) in zip(bs, cs) {
            if let scalar = UnicodeScalar(c) {
                encoder[b] = String(scalar)
            }
        }
        return encoder
    }

    private func byteLevelEncode(_ token: String, encoder: [UInt8: String]) -> [Character] {
        var output = ""
        for byte in token.utf8 {
            output += encoder[byte] ?? ""
        }
        return Array(output)
    }

    private func loadNpyInt32(named fileName: String) -> [Int32]? {
        let fileManager = FileManager.default
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        let documentPath = documentsURL?.appendingPathComponent(fileName)
        let bundlePath = Bundle.main.url(forResource: fileName.replacingOccurrences(of: ".npy", with: ""), withExtension: "npy")

        let urlToUse: URL?
        if let documentPath, fileManager.fileExists(atPath: documentPath.path) {
            urlToUse = documentPath
        } else {
            urlToUse = bundlePath
        }

        guard let url = urlToUse else {
            appendLog("Missing \(fileName). Using fallback tokenizer.")
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            let (values, shape) = try parseNpyInt32(data: data)
            if shape.count == 2, shape[0] == 1 {
                appendLog("Loaded \(fileName) with shape [\(shape[0]), \(shape[1])]")
            } else {
                appendLog("Unexpected shape for \(fileName): \(shape)")
            }
            return values
        } catch {
            appendLog("Failed to load \(fileName): \(error.localizedDescription)")
            return nil
        }
    }

    private func parseNpyInt32(data: Data) throws -> ([Int32], [Int]) {
        guard data.count > 10 else { throw NSError(domain: "NPY", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid npy data"]) }
        let magic = data.subdata(in: 0..<6)
        guard magic == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw NSError(domain: "NPY", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid npy header"])
        }
        let major = data[6]
        let headerLenOffset = 8
        let headerLen: Int
        if major == 1 {
            headerLen = Int(data.withUnsafeBytes { $0.load(fromByteOffset: headerLenOffset, as: UInt16.self).littleEndian })
        } else {
            headerLen = Int(data.withUnsafeBytes { $0.load(fromByteOffset: headerLenOffset, as: UInt32.self).littleEndian })
        }
        let headerStart = major == 1 ? 10 : 12
        let headerEnd = headerStart + headerLen
        guard headerEnd <= data.count else { throw NSError(domain: "NPY", code: -3, userInfo: [NSLocalizedDescriptionKey: "Invalid npy header length"]) }
        let headerData = data.subdata(in: headerStart..<headerEnd)
        let header = String(data: headerData, encoding: .isoLatin1) ?? ""

        guard header.contains("'descr': '<i4'") else {
            throw NSError(domain: "NPY", code: -4, userInfo: [NSLocalizedDescriptionKey: "Unsupported dtype (expected int32)"])
        }

        let shape = parseNpyShape(from: header)
        let dataStart = headerEnd
        let remaining = data.count - dataStart
        guard remaining % 4 == 0 else { throw NSError(domain: "NPY", code: -5, userInfo: [NSLocalizedDescriptionKey: "Invalid data length"]) }

        let intCount = remaining / 4
        let values: [Int32] = data.subdata(in: dataStart..<data.count).withUnsafeBytes { rawBuffer in
            let buffer = rawBuffer.bindMemory(to: Int32.self)
            return Array(buffer)
        }

        guard values.count == intCount else {
            throw NSError(domain: "NPY", code: -6, userInfo: [NSLocalizedDescriptionKey: "Failed to parse npy data"])
        }

        return (values, shape)
    }

    private func parseNpyShape(from header: String) -> [Int] {
        guard let shapeRange = header.range(of: "shape") else { return [] }
        let afterShape = header[shapeRange.upperBound...]
        guard let startParen = afterShape.firstIndex(of: "("),
              let endParen = afterShape.firstIndex(of: ")") else { return [] }
        let shapeContents = afterShape[afterShape.index(after: startParen)..<endParen]
        return shapeContents
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)) }
    }

    private func decodeWavToMonoFloat32(audioData: Data, targetSampleRate: Int) throws -> [Float] {
        struct WAVHeader {
            var sampleRate: Int = 0
            var channels: Int = 0
            var bitsPerSample: Int = 0
            var dataStart: Int = 0
            var dataSize: Int = 0
        }

        func readUInt32LE(_ data: Data, _ offset: Int) -> UInt32 {
            let value = data.subdata(in: offset..<offset+4)
            return value.withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian
        }

        func readUInt16LE(_ data: Data, _ offset: Int) -> UInt16 {
            let value = data.subdata(in: offset..<offset+2)
            return value.withUnsafeBytes { $0.load(as: UInt16.self) }.littleEndian
        }

        guard audioData.count >= 44 else {
            throw NSError(domain: "Audio", code: -3, userInfo: [NSLocalizedDescriptionKey: "Invalid WAV data"])
        }

        var header = WAVHeader()
        var offset = 12 // Skip RIFF header
        while offset + 8 <= audioData.count {
            let chunkID = String(data: audioData.subdata(in: offset..<offset+4), encoding: .ascii) ?? ""
            let chunkSize = Int(readUInt32LE(audioData, offset + 4))
            let chunkDataStart = offset + 8

            if chunkID == "fmt " {
                let audioFormat = Int(readUInt16LE(audioData, chunkDataStart))
                header.channels = Int(readUInt16LE(audioData, chunkDataStart + 2))
                header.sampleRate = Int(readUInt32LE(audioData, chunkDataStart + 4))
                header.bitsPerSample = Int(readUInt16LE(audioData, chunkDataStart + 14))
                guard audioFormat == 1 else {
                    throw NSError(domain: "Audio", code: -4, userInfo: [NSLocalizedDescriptionKey: "Unsupported WAV format (only PCM supported)"])
                }
            } else if chunkID == "data" {
                header.dataStart = chunkDataStart
                header.dataSize = chunkSize
                break
            }

            offset = chunkDataStart + chunkSize
        }

        guard header.dataSize > 0, header.dataStart + header.dataSize <= audioData.count else {
            throw NSError(domain: "Audio", code: -5, userInfo: [NSLocalizedDescriptionKey: "WAV data chunk not found"])
        }

        guard header.bitsPerSample == 16 else {
            throw NSError(domain: "Audio", code: -6, userInfo: [NSLocalizedDescriptionKey: "Unsupported WAV bit depth (expected 16-bit PCM)"])
        }

        let dataChunk = audioData.subdata(in: header.dataStart..<(header.dataStart + header.dataSize))
        let int16Samples = dataChunk.withUnsafeBytes { rawBuffer -> [Int16] in
            let buffer = rawBuffer.bindMemory(to: Int16.self)
            return Array(buffer)
        }

        let channels = max(header.channels, 1)
        var monoSamples = [Float]()
        monoSamples.reserveCapacity(int16Samples.count / channels)
        for i in stride(from: 0, to: int16Samples.count, by: channels) {
            let sample = Float(int16Samples[i]) / 32768.0
            monoSamples.append(sample)
        }

        if header.sampleRate == targetSampleRate {
            return monoSamples
        }

        // Simple linear resample to targetSampleRate
        let ratio = Double(targetSampleRate) / Double(header.sampleRate)
        let newCount = Int(Double(monoSamples.count) * ratio)
        var resampled = [Float]()
        resampled.reserveCapacity(newCount)
        for i in 0..<newCount {
            let srcIndex = Double(i) / ratio
            let idx = Int(srcIndex)
            let frac = srcIndex - Double(idx)
            if idx + 1 < monoSamples.count {
                let v = monoSamples[idx] * Float(1 - frac) + monoSamples[idx + 1] * Float(frac)
                resampled.append(v)
            } else if let last = monoSamples.last {
                resampled.append(last)
            }
        }
        return resampled
    }

    func resetModelCache() async {
        await MainActor.run {
            statusMessage = "Resetting model cache..."
        }
        appendLog("Resetting model cache")

        let fileManager = FileManager.default
        guard let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            appendLog("Failed to locate Documents directory")
            return
        }

        do {
            let contents = try fileManager.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil)
            for item in contents {
                var isDirectory: ObjCBool = false
                if fileManager.fileExists(atPath: item.path, isDirectory: &isDirectory), isDirectory.boolValue {
                    let subitems = (try? fileManager.contentsOfDirectory(at: item, includingPropertiesForKeys: nil)) ?? []
                    let hasZeticArtifacts = subitems.contains { $0.lastPathComponent.hasPrefix("ZETIC_MLANGE_TARGET_") }
                    if hasZeticArtifacts {
                        try fileManager.removeItem(at: item)
                        appendLog("Deleted model cache folder: \(item.lastPathComponent)")
                    }
                }
            }
        } catch {
            appendLog("Failed to reset model cache: \(error.localizedDescription)")
        }

        await MainActor.run {
            isInitialized = false
            backboneModel = nil
            encoderModel = nil
            decoderModel = nil
            statusMessage = "Model cache cleared. Reinitializing..."
        }

        await initializeModels()
    }

    func playAudio(data: Data) {
        do {
            audioPlayer = try AVAudioPlayer(data: data)
            audioPlayer?.play()
        } catch {
            errorMessage = "Failed to play audio: \(error.localizedDescription)"
        }
    }

    func stopAudio() {
        audioPlayer?.stop()
    }

    func appendLog(_ message: String) {
        Task { @MainActor in
            let timestamp = ISO8601DateFormatter().string(from: Date())
            let line = "[\(timestamp)] \(message)"
            print(line)
            logLines.append(line)
            if logLines.count > 500 {
                logLines.removeFirst(logLines.count - 500)
            }
        }
    }
}
