import SwiftUI

/// Renders MedGemma's lightly-structured Markdown (## headings, _italic_ disclaimer,
/// plain body) and shows a blinking caret while still streaming.
struct StreamingText: View {
    let text: String
    var isStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block.kind {
                case .heading:
                    HStack(spacing: 8) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Theme.brandGradient)
                            .frame(width: 3, height: 15)
                        Text(block.text)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(Theme.ink)
                    }
                case .italic:
                    Text(block.text)
                        .font(.system(size: 12.5))
                        .italic()
                        .foregroundStyle(Theme.inkFaint)
                case .body:
                    Text(block.attributed)
                        .font(.system(size: 14.5))
                        .foregroundStyle(Theme.inkSoft)
                        .lineSpacing(3)
                }
            }
            if isStreaming { BlinkingCaret() }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private struct Block { enum Kind { case heading, body, italic }; let kind: Kind; let text: String
        var attributed: AttributedString {
            (try? AttributedString(markdown: text)) ?? AttributedString(text)
        }
    }

    private var blocks: [Block] {
        text.split(separator: "\n", omittingEmptySubsequences: true).map { rawLine in
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("## ") {
                return Block(kind: .heading, text: String(line.dropFirst(3)))
            }
            if line.hasPrefix("#") {
                return Block(kind: .heading, text: String(line.drop(while: { $0 == "#" || $0 == " " })))
            }
            if line.hasPrefix("_") && line.hasSuffix("_") && line.count > 2 {
                return Block(kind: .italic, text: String(line.dropFirst().dropLast()))
            }
            return Block(kind: .body, text: line)
        }
    }
}

/// Soft pulsing caret shown during generation.
private struct BlinkingCaret: View {
    @State private var on = true
    var body: some View {
        RoundedRectangle(cornerRadius: 1.5)
            .fill(Theme.accent)
            .frame(width: 8, height: 16)
            .opacity(on ? 1 : 0.15)
            .shadow(color: Theme.accent.opacity(0.7), radius: 4)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true)) { on.toggle() }
            }
    }
}
