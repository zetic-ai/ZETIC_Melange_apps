import SwiftUI

/// Summary card shown after a guided 30-second measurement.
struct ReportView: View {
    let report: MeasurementReport
    let onDone: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 22) {
                    VStack(spacing: 6) {
                        Text("Measurement complete")
                            .font(.system(size: 22, weight: .bold, design: .rounded))
                            .foregroundStyle(Theme.textPrimary)
                        Text(Theme.qualityLabel(report.avgQuality))
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundStyle(Theme.quality(report.avgQuality))
                    }

                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Image(systemName: "heart.fill").foregroundStyle(Theme.accent).font(.system(size: 28))
                        Text("\(report.avgBPM)")
                            .font(.system(size: 72, weight: .bold, design: .rounded))
                            .monospacedDigit()
                            .foregroundStyle(Theme.textPrimary)
                        Text("BPM").font(.system(size: 18, weight: .semibold, design: .rounded))
                            .foregroundStyle(Theme.textSecondary)
                    }

                    sparkline

                    HStack(spacing: 12) {
                        stat("Min", report.minBPM)
                        stat("Average", report.avgBPM)
                        stat("Max", report.maxBPM)
                    }

                    Text("Not a medical device. For demonstration only.")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.textSecondary)
                }
                .padding(.horizontal, 24)
                .padding(.top, 28)
                .padding(.bottom, 12)
            }

            Button(action: onDone) {
                Text("Done")
                    .font(.system(size: 17, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Theme.accent, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 20)
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        .background(Theme.background)
    }

    private var sparkline: some View {
        Card {
            WaveformChart(samples: report.series.map { Float($0) }, color: Theme.accent)
                .frame(height: 60)
                .padding(14)
        }
        .frame(height: 88)
    }

    private func stat(_ label: String, _ value: Int) -> some View {
        VStack(spacing: 4) {
            Text("\(value)")
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(Theme.textPrimary)
            Text(label)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundStyle(Theme.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(Theme.accentSoft, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}
