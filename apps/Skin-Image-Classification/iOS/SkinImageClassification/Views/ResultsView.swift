import SwiftUI

/// The analysis result: on-device classifier verdict + ranked distribution + curated
/// per-condition guidance, all over the glass aesthetic.
struct ResultsView: View {
    @EnvironmentObject private var vm: DiagnosisViewModel

    var body: some View {
        VStack(spacing: 0) {
            topBar

            if let c = vm.classification {
                ScrollView {
                    VStack(spacing: 16) {
                        verdictCard(c)
                        ClassDistributionBars(scores: c.ranked).glassCard(cornerRadius: 22)
                        guidanceCard(c.topClass)
                        DisclaimerBanner()
                        Color.clear.frame(height: 8)
                    }
                    .padding(.horizontal, 18)
                    .padding(.top, 8)
                }
            } else if case .failed(let msg) = vm.analysis {
                // Failure BEFORE any classification — surface it instead of spinning forever.
                earlyFailureState(msg)
            } else {
                analyzingState
            }
        }
        .padding(.top, 10)
    }

    // MARK: Top bar

    private var topBar: some View {
        HStack(spacing: 12) {
            Button { vm.reset() } label: {
                Image(systemName: "chevron.left")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Theme.ink)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }
            if let img = vm.image {
                Image(uiImage: img)
                    .resizable().scaledToFill()
                    .frame(width: 38, height: 38)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
            VStack(alignment: .leading, spacing: 1) {
                Text("Analysis").font(.system(size: 16, weight: .bold)).foregroundStyle(Theme.ink)
                Text("Fully on-device").font(.system(size: 11)).foregroundStyle(Theme.inkFaint)
            }
            Spacer()
            Button { vm.reset() } label: {
                Text("New").font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.accent)
            }
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 8)
    }

    // MARK: Verdict

    private func verdictCard(_ c: Classification) -> some View {
        VStack(spacing: 16) {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    SeverityBadge(skinClass: c.topClass)
                    Text(c.topClass.title)
                        .font(.system(size: 23, weight: .bold))
                        .foregroundStyle(Theme.ink)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(c.topClass.blurb)
                        .font(.system(size: 13.5))
                        .foregroundStyle(Theme.inkSoft)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                ConfidenceRing(value: c.confidence, tint: c.topClass.tint, size: 116)
            }

            if c.isLowConfidence {
                Label("Low confidence — treat this as a rough hint only.", systemImage: "questionmark.circle")
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(Theme.amber)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 6) {
                Image(systemName: "bolt.fill").font(.system(size: 10))
                Text("Classified on-device in \(Int(c.latencyMs)) ms")
                    .font(Theme.mono(11, .medium))
            }
            .foregroundStyle(Theme.inkFaint)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .glassCard(cornerRadius: 26)
    }

    // MARK: Guidance (curated, per condition)

    private func guidanceCard(_ cls: SkinClass) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            guidanceSection(icon: "doc.text.magnifyingglass", title: "What this result may suggest") {
                Text(cls.whatItMeans)
                    .font(.system(size: 14)).foregroundStyle(Theme.inkSoft).lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            guidanceSection(icon: "heart.text.square", title: "General self-care") {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(cls.selfCare, id: \.self) { tip in
                        HStack(alignment: .top, spacing: 8) {
                            Circle().fill(Theme.accent).frame(width: 5, height: 5).padding(.top, 6)
                            Text(tip).font(.system(size: 14)).foregroundStyle(Theme.inkSoft)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
            guidanceSection(icon: "stethoscope", title: "When to seek medical care") {
                Text(cls.whenToSeeDoctor)
                    .font(.system(size: 14)).foregroundStyle(Theme.inkSoft).lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(cornerRadius: 22)
    }

    private func guidanceSection<Content: View>(icon: String, title: String,
                                                @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                RoundedRectangle(cornerRadius: 2).fill(Theme.brandGradient).frame(width: 3, height: 14)
                Image(systemName: icon).font(.system(size: 12)).foregroundStyle(Theme.accent2)
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.ink)
            }
            content()
        }
    }

    // MARK: Classifying placeholder

    private var analyzingState: some View {
        VStack(spacing: 22) {
            Spacer()
            if let img = vm.image {
                Image(uiImage: img)
                    .resizable().scaledToFill()
                    .frame(width: 180, height: 180)
                    .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                    .overlay(ScanOverlay())
                    .overlay(
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .strokeBorder(Theme.accent.opacity(0.5), lineWidth: 1.5)
                    )
            }
            VStack(spacing: 6) {
                Text("Analyzing on-device")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.ink)
                Text("Running the skin vision model…")
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.inkSoft)
            }
            Spacer()
            Spacer()
        }
    }

    private func earlyFailureState(_ msg: String) -> some View {
        VStack(spacing: 18) {
            Spacer()
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40)).foregroundStyle(Theme.coral)
            Text("Analysis couldn't finish")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Theme.ink)
            Text(msg)
                .font(.system(size: 13.5))
                .foregroundStyle(Theme.inkSoft)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 30)
            HStack(spacing: 12) {
                SecondaryButton(title: "Back", icon: "chevron.left") { vm.reset() }
                if let img = vm.image {
                    PrimaryButton(title: "Retry", icon: "arrow.clockwise") { vm.analyze(img) }
                }
            }
            .padding(.horizontal, 40)
            Spacer(); Spacer()
        }
    }
}

/// Risk pill colored by severity.
struct SeverityBadge: View {
    let skinClass: SkinClass
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(skinClass.tint).frame(width: 7, height: 7)
                .shadow(color: skinClass.tint, radius: 4)
            Text(skinClass.severityLabel)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(skinClass.tint)
                .textCase(.uppercase).tracking(0.8)
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(skinClass.tint.opacity(0.12), in: Capsule())
        .overlay(Capsule().strokeBorder(skinClass.tint.opacity(0.3), lineWidth: 1))
    }
}

/// Sweeping scan line over the thumbnail while classifying.
private struct ScanOverlay: View {
    @State private var y: CGFloat = -90
    var body: some View {
        Rectangle()
            .fill(LinearGradient(colors: [.clear, Theme.accent.opacity(0.8), .clear],
                                 startPoint: .top, endPoint: .bottom))
            .frame(height: 36)
            .offset(y: y)
            .onAppear {
                withAnimation(.easeInOut(duration: 1.3).repeatForever(autoreverses: true)) { y = 90 }
            }
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
    }
}
