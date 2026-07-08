import SwiftUI

/// Photo capture screen: take or choose a skin photo, preview it, then analyze.
struct CaptureView: View {
    @EnvironmentObject private var vm: DiagnosisViewModel
    @State private var picked: UIImage?
    @State private var showCamera = false
    @State private var showLibrary = false

    var body: some View {
        VStack(spacing: 24) {
            header

            Spacer(minLength: 8)

            previewArea

            Spacer(minLength: 8)

            actions

            DisclaimerBanner().padding(.horizontal, 4)
        }
        .padding(.horizontal, 22)
        .padding(.top, 18)
        .padding(.bottom, 16)
        .sheet(isPresented: $showCamera) {
            CameraPicker { picked = $0 }.ignoresSafeArea()
        }
        .sheet(isPresented: $showLibrary) {
            LibraryPicker { picked = $0 }.ignoresSafeArea()
        }
    }

    private var header: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .font(.system(size: 14))
                    .foregroundStyle(Theme.accent)
                Text("SKIN CLASSIFIER")
                    .font(Theme.mono(13, .semibold))
                    .tracking(3)
                    .foregroundStyle(Theme.brandGradient)
            }
            Text("On-device skin analysis")
                .font(.system(size: 24, weight: .bold))
                .foregroundStyle(Theme.ink)
            Text("Take or choose a clear, close-up photo of the skin area.")
                .font(.system(size: 13.5))
                .foregroundStyle(Theme.inkSoft)
                .multilineTextAlignment(.center)
        }
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(Color.white.opacity(0.04))
                .overlay(
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .strokeBorder(Theme.hairline, lineWidth: 1)
                )

            if let img = picked {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFill()
                    .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 28, style: .continuous)
                            .strokeBorder(Theme.accent.opacity(0.4), lineWidth: 1.5)
                    )
                    .overlay(alignment: .topTrailing) {
                        Button { picked = nil } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundStyle(.white)
                                .padding(9)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                        .padding(12)
                    }
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 44, weight: .light))
                        .foregroundStyle(Theme.inkFaint)
                    Text("No photo selected")
                        .font(.system(size: 14))
                        .foregroundStyle(Theme.inkFaint)
                }
            }
        }
        .frame(height: 320)
    }

    @ViewBuilder private var actions: some View {
        if picked != nil {
            PrimaryButton(title: "Analyze skin", icon: "waveform.path.ecg") {
                if let img = picked { vm.analyze(img) }
            }
        } else {
            HStack(spacing: 12) {
                if CameraPicker.isAvailable {
                    SecondaryButton(title: "Camera", icon: "camera.fill") { showCamera = true }
                }
                SecondaryButton(title: "Library", icon: "photo.fill.on.rectangle.fill") { showLibrary = true }
            }
        }
    }
}

// MARK: - Buttons

struct PrimaryButton: View {
    let title: String
    var icon: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 9) {
                if let icon { Image(systemName: icon).font(.system(size: 16, weight: .semibold)) }
                Text(title).font(.system(size: 16, weight: .semibold))
            }
            .foregroundStyle(Color(red: 0.03, green: 0.05, blue: 0.08))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(Theme.brandGradient, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .shadow(color: Theme.accent.opacity(0.4), radius: 14, y: 6)
        }
    }
}

struct SecondaryButton: View {
    let title: String
    var icon: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if let icon { Image(systemName: icon).font(.system(size: 15, weight: .medium)) }
                Text(title).font(.system(size: 15, weight: .semibold))
            }
            .foregroundStyle(Theme.ink)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(Theme.hairline, lineWidth: 1)
            )
        }
    }
}
