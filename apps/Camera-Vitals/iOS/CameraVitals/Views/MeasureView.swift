import SwiftUI

/// The live measurement screen: camera + face lock, heart rate, waveform, and a guided scan.
struct MeasureView: View {
    @ObservedObject var vm: VitalsViewModel

    private var isWarmup: Bool {
        if case .warmup = vm.state { return true }
        return false
    }

    private var warmupFraction: Double {
        vm.warmupProgress
    }

    var body: some View {
        VStack(spacing: 16) {
            header
            cameraCard
            bpmCard
            waveformCard
            Spacer(minLength: 0)
            measureControl
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
        .padding(.bottom, 16)
    }

    // MARK: - Sections

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Camera Vitals")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundStyle(Theme.textPrimary)
                Text("Powered by ZETIC Melange")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer()
            OnDeviceHUD(latencyMs: vm.latencyMs)
        }
    }

    private var cameraCard: some View {
        Card {
            ZStack {
                CameraPreview(session: vm.camera.session)

                FaceLockOverlay(
                    faceBox: vm.faceBox,
                    bufferSize: vm.bufferSize,
                    quality: vm.quality,
                    faceFound: vm.faceFound
                )

                VStack {
                    if isWarmup {
                        warmupBar
                    }
                    Spacer()
                    HStack {
                        SignalQualityBadge(quality: vm.quality, faceFound: vm.faceFound, lowLight: vm.lowLight)
                        Spacer()
                    }
                    .padding(12)
                }
            }
        }
        .frame(height: 340)
        .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
    }

    private var warmupBar: some View {
        VStack(spacing: 4) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.35)).frame(height: 5)
                    Capsule().fill(Theme.accent)
                        .frame(width: geo.size.width * warmupFraction, height: 5)
                        .animation(.easeOut(duration: 0.2), value: warmupFraction)
                }
            }
            .frame(height: 5)
            Text("Stabilizing… keep still")
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(.white)
                .shadow(radius: 2)
        }
        .padding(12)
    }

    private var bpmCard: some View {
        Card {
            HStack {
                BPMReadout(bpm: displayBPM, quality: vm.quality)
                Spacer()
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 18)
        }
    }

    private var waveformCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 8) {
                Text("Pulse waveform")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(Theme.textSecondary)
                WaveformChart(samples: vm.waveform)
                    .frame(height: 64)
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 16)
        }
    }

    private var measureControl: some View {
        Group {
            if vm.isMeasuring {
                VStack(spacing: 10) {
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Theme.accentSoft).frame(height: 8)
                            Capsule().fill(Theme.accent)
                                .frame(width: geo.size.width * vm.measureProgress, height: 8)
                        }
                    }
                    .frame(height: 8)
                    HStack {
                        Text("Measuring… hold still")
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundStyle(Theme.textSecondary)
                        Spacer()
                        Text("\(Int((1 - vm.measureProgress) * AppConfig.measureDuration))s")
                            .font(.system(size: 14, weight: .bold, design: .rounded))
                            .monospacedDigit()
                            .foregroundStyle(Theme.accent)
                        Button("Cancel") { vm.cancelMeasurement() }
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundStyle(Theme.poor)
                    }
                }
                .padding(.horizontal, 4)
            } else {
                Button { vm.startMeasurement() } label: {
                    Text("Measure 30s")
                        .font(.system(size: 17, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(canMeasure ? Theme.accent : Theme.textSecondary.opacity(0.4),
                                    in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                }
                .disabled(!canMeasure)
            }
        }
    }

    // MARK: - Derived

    private var canMeasure: Bool {
        if case .live = vm.state { return vm.faceFound }
        return false
    }

    private var displayBPM: Double? {
        if case .warmup = vm.state { return nil }
        return vm.bpm
    }
}
