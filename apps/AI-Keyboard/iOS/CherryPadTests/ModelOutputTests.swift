import XCTest
import ZeticMLange
@testable import CherryPad

/// De-risk: does LFM2.5-350M stay coherent with a small context (nCtx=512)?
/// A small nCtx is the main lever to keep the keyboard under the jetsam limit, but
/// an explicit init previously degraded Qwen — so verify LFM output here. Device-only.
final class ModelOutputTests: XCTestCase {

    func testLFM_nCtx512() throws {
        let model = try ZeticMLangeLLMModel(
            personalKey: ZeticConfig.personalKey,
            name: "Steve/LFM2.5_350M",
            version: 1,
            modelMode: LLMModelMode.RUN_ACCURACY,
            initOption: LLMInitOption(kvCacheCleanupPolicy: .CLEAN_UP_ON_FULL, nCtx: 512),
            onDownload: { p in if p > 0, p < 1 { print("⬇️ \(Int(p*100))%") } }
        )
        let cases: [(String, String)] = [
            ("REWRITE", Prompts.build(task: .rewrite, text: "hi i like your company. i want work with you. pls check my cv.",
                tone: .professional, stance: nil, targetLanguage: nil)),
            ("REPLY", Prompts.build(task: .reply, text: "Want to grab coffee sometime this week?",
                tone: nil, stance: .agreeable, targetLanguage: nil)),
            ("TRANSLATE_KO", Prompts.build(task: .translate, text: "Good to see you. Let's catch up soon.",
                tone: nil, stance: nil, targetLanguage: "Korean")),
            ("GRAMMAR", Prompts.build(task: .grammar, text: "he go to school yesterday and dont did his homework",
                tone: nil, stance: nil, targetLanguage: nil)),
        ]
        for (name, prompt) in cases {
            try? model.cleanUp()
            _ = try model.run(prompt)
            var raw = ""; var n = 0
            while n < 64 {   // keyboard-style hard cap
                let r = model.waitForNextToken()
                if r.token.isEmpty || r.isFinished { break }
                raw += r.token; n += 1
            }
            try? model.cleanUp()
            print("@@@ \(name) (\(n)t): \(LLMOutput.sanitize(raw).replacingOccurrences(of: "\n", with: " ⏎ "))")
        }
        model.forceDeinit()
    }
}
