import Foundation
import SwiftUI
import Combine

/// Drives the guided multimodal check-in: a styled Berry avatar asks open-ended
/// questions while the front camera streams face emotion (Melange FER) and the mic
/// records each answer (Melange voice SER + on-device transcript). At the end it
/// fuses everything into an explainable, non-diagnostic `ScreeningReport`.
///
/// State machine: idle → intro → question(i)… → analyzing → insights(report).
@MainActor
final class CheckInSession: ObservableObject {
    enum Phase: Equatable {
        case idle
        case intro
        case question(Int)
        case analyzing
        case insights(ScreeningReport)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var elapsed: Double = 0      // seconds into the current question
    /// The current question revealed word-by-word as TTS speaks it. While speaking
    /// this grows from "" to the full question; once recording starts it is the full text.
    @Published private(set) var spokenText: String = ""
    /// True while the question is being read aloud (before recording the answer).
    @Published private(set) var isSpeaking: Bool = false

    let camera = CameraController()
    private let voice: EmotionModel
    private let face: FaceEmotionModel
    private let recorder = AudioRecorder()
    private let sessionAudio = SessionAudio()
    private let transcriber = SpeechTranscriber()
    private let speaker = QuestionSpeaker()

    private var timer: AnyCancellable?
    private var voiceClips: [[Float]] = []
    private let assembly = DispatchQueue(label: "aiberry.checkin.assembly")

    private let questions = AppConfig.CheckIn.questions

    init(voice: EmotionModel, face: FaceEmotionModel) {
        self.voice = voice
        self.face = face
        recorder.ownsSession = false                        // SessionAudio owns the session
        camera.onFrame = { [weak self] buffer in self?.face.ingest(buffer) }
    }

    // MARK: derived UI state

    var questionIndex: Int { if case .question(let i) = phase { return i } else { return 0 } }
    var totalQuestions: Int { questions.count }
    var currentQuestion: String { questions[min(questionIndex, questions.count - 1)] }
    var canAdvance: Bool { elapsed >= AppConfig.CheckIn.minSeconds }
    var countdownProgress: Double { min(1, elapsed / AppConfig.CheckIn.maxSeconds) }
    var isRecording: Bool { recorder.isRecording }
    var micLevel: Float { recorder.level }
    /// Live top emotion from the face stream, for the avatar expression.
    var liveEmotion: String? { face.liveTop?.label }

    // MARK: flow

    func showIntro() { phase = .intro }

    func begin() {
        voiceClips = []
        face.reset()
        sessionAudio.begin()
        camera.start()
        phase = .question(0)
        startQuestion()
    }

    /// User taps "Next question" (only enabled after `minSeconds`).
    func advance() {
        guard canAdvance, recorder.isRecording else { return }
        recorder.stop()                                     // completion → finishQuestion
    }

    /// User taps "End" — abandon the session.
    func cancel() {
        teardownCapture()
        phase = .idle
    }

    func restart() {
        phase = .idle
    }

    private func startQuestion() {
        elapsed = 0
        isSpeaking = true
        spokenText = ""
        // Read the question aloud, revealing it word-by-word, then start recording.
        speaker.speak(currentQuestion,
                      onWord: { [weak self] prefix in self?.spokenText = prefix },
                      onFinish: { [weak self] in self?.beginRecording() })
    }

    /// Called once the question has finished being spoken — start capturing the answer.
    private func beginRecording() {
        guard case .question = phase else { return }   // ignore if the user ended/advanced
        isSpeaking = false
        spokenText = currentQuestion
        elapsed = 0
        startTimer()
        recorder.record(autoStopSeconds: AppConfig.CheckIn.maxSeconds) { [weak self] samples in
            self?.finishQuestion(samples)
        }
    }

    private func finishQuestion(_ samples: [Float]) {
        stopTimer()
        voiceClips.append(samples)
        let next = questionIndex + 1
        if next < questions.count {
            phase = .question(next)
            startQuestion()
        } else {
            analyze()
        }
    }

    private func analyze() {
        stopTimer()
        camera.stop()
        sessionAudio.end()
        phase = .analyzing

        let clips = voiceClips
        let concat = clips.flatMap { $0 }
        let voiced = AudioUtils.voicedFraction(concat)

        var faceDist: [Float] = []
        var faceFrames = 0
        var voiceProbs: [Float] = []
        var transcripts = [String](repeating: "", count: clips.count)

        let group = DispatchGroup()

        group.enter()
        face.finalize { dist, frames in
            self.assembly.async { faceDist = dist; faceFrames = frames; group.leave() }
        }

        group.enter()
        voice.analyze(concat) { probs in
            self.assembly.async { voiceProbs = probs; group.leave() }
        }

        group.enter()
        transcriber.transcribeAll(clips) { texts in
            self.assembly.async { transcripts = texts; group.leave() }
        }

        group.notify(queue: .main) { [weak self] in
            guard let self else { return }
            let pairs = self.questions.prefix(clips.count).enumerated().map {
                QAPair(question: $0.element, answer: transcripts[$0.offset])
            }
            let report = FusionEngine.fuse(face: faceDist, faceFrames: faceFrames,
                                           voice: voiceProbs, voicedFraction: voiced,
                                           transcript: Array(pairs))
            Haptics.success()
            self.phase = .insights(report)
        }
    }

    // MARK: timer / teardown

    private func startTimer() {
        timer = Timer.publish(every: 0.1, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                guard let self else { return }
                self.elapsed += 0.1
            }
    }

    private func stopTimer() { timer?.cancel(); timer = nil }

    private func teardownCapture() {
        stopTimer()
        speaker.cancel()
        isSpeaking = false
        if recorder.isRecording { recorder.stop() }
        camera.stop()
        sessionAudio.end()
        face.reset()
    }
}
