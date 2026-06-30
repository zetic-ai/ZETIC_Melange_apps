import XCTest
import ZeticMLange
@testable import CherryPad

final class ModelOutputTests: XCTestCase {

    /// Verifies the keyboard↔app data path: App Group request/result round-trip and
    /// deep-link id parsing. This is the shared contract that drives the handoff.
    func testHandoffRoundTrip() throws {
        HandoffStore.clearRequest()
        HandoffStore.clearResult()

        let req = HandoffRequest(task: .rewrite, text: "hello world",
                                 tone: .professional, stance: nil, targetLanguage: nil)
        HandoffStore.writeRequest(req)

        let read = HandoffStore.readRequest()
        XCTAssertEqual(read?.id, req.id)
        XCTAssertEqual(read?.text, "hello world")
        XCTAssertEqual(read?.task, .rewrite)
        XCTAssertEqual(read?.tone, .professional)

        // Deep link carries the request id and parses back.
        let url = try XCTUnwrap(DeepLink.process(id: req.id))
        XCTAssertEqual(url.scheme, "cherrypad")
        XCTAssertEqual(DeepLink.requestID(from: url), req.id)

        // Result round-trip.
        HandoffStore.writeResult(HandoffResult(requestID: req.id, text: "Hello, world."))
        XCTAssertEqual(HandoffStore.readResult()?.text, "Hello, world.")

        // Clearing works.
        HandoffStore.clearRequest()
        HandoffStore.clearResult()
        XCTAssertNil(HandoffStore.readRequest())
        XCTAssertNil(HandoffStore.readResult())
    }

    /// Default-tier model smoke test (Qwen3-0.6B) — keep as a regression guard.
    func testDefaultTierProducesCoherentOutput() throws {
        ZeticConfig.quality = .qwen
        let model = try ZeticMLangeLLMModel(
            personalKey: ZeticConfig.personalKey, name: ZeticConfig.Quality.qwen.modelName,
            version: ZeticConfig.modelVersion, onDownload: { _ in })
        let cases: [(String, String)] = [
            ("REWRITE", Prompts.build(task: .rewrite, text: "hi i like your company. pls check my cv.",
                tone: .professional, stance: nil, targetLanguage: nil)),
            ("GRAMMAR", Prompts.build(task: .grammar, text: "he go to school yesterday and dont did his homework",
                tone: nil, stance: nil, targetLanguage: nil)),
        ]
        for (name, prompt) in cases {
            try? model.cleanUp()
            _ = try model.run(prompt)
            var raw = ""; var n = 0
            while n < 200 {
                let r = model.waitForNextToken()
                if r.token.isEmpty || r.isFinished { break }
                raw += r.token; n += 1
            }
            try? model.cleanUp()
            let clean = LLMOutput.sanitize(raw)
            print("@@@ \(name): \(clean.replacingOccurrences(of: "\n", with: " ⏎ "))")
            XCTAssertFalse(clean.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        model.forceDeinit()
    }
}
