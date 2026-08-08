import Foundation

final class EspeakPhonemizer {
    static let shared = EspeakPhonemizer()

    private var initialized = false

    private init() {}

    func phonemize(_ text: String) -> String {
#if ESPEAK_AVAILABLE
        guard ensureInitialized() else {
            print("[EspeakPhonemizer] Initialization failed; falling back to raw text.")
            return text
        }

        let cString = Array(text.utf8CString)
        return cString.withUnsafeBufferPointer { buffer in
            var ptr = buffer.baseAddress
            let mode = Int32(espeakCHARS_UTF8)
            let phonemeMode = Int32(espeakPHONEMES)
            guard let result = espeak_TextToPhonemes(&ptr, mode, phonemeMode) else {
                return text
            }
            return String(cString: result)
        }
#else
        return text
#endif
    }

#if ESPEAK_AVAILABLE
    private func ensureInitialized() -> Bool {
        if initialized {
            return true
        }

        guard let dataPath = Bundle.main.path(forResource: "espeak-ng-data", ofType: nil) else {
            print("[EspeakPhonemizer] espeak-ng-data not found in bundle.")
            return false
        }

        dataPath.withCString { espeak_SetPath($0) }
        let output = Int32(espeakOUTPUT_SYNCHRONOUS)
        let sampleRate = espeak_Initialize(output, 0, dataPath, 0)
        if sampleRate <= 0 {
            print("[EspeakPhonemizer] espeak_Initialize failed.")
            return false
        }

        _ = espeak_SetVoiceByName("en-us")
        initialized = true
        return true
    }
#endif
}

#if ESPEAK_AVAILABLE
// espeak-ng C API bindings
@_silgen_name("espeak_Initialize")
private func espeak_Initialize(_ output: Int32, _ bufferLength: Int32, _ path: UnsafePointer<CChar>?, _ options: Int32) -> Int32

@_silgen_name("espeak_SetPath")
private func espeak_SetPath(_ path: UnsafePointer<CChar>)

@_silgen_name("espeak_SetVoiceByName")
private func espeak_SetVoiceByName(_ name: UnsafePointer<CChar>?) -> Int32

@_silgen_name("espeak_TextToPhonemes")
private func espeak_TextToPhonemes(_ textPtr: UnsafeMutablePointer<UnsafePointer<CChar>?>?, _ textMode: Int32, _ phonemeMode: Int32) -> UnsafePointer<CChar>?

private let espeakOUTPUT_SYNCHRONOUS = 0
private let espeakCHARS_UTF8 = 1
private let espeakPHONEMES = 0x02
#endif

