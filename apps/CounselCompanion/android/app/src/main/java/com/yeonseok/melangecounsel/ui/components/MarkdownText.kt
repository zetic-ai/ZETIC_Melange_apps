package com.yeonseok.melangecounsel.ui.components

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow

@Composable
fun MarkdownText(
    text: String,
    color: Color,
    maxLines: Int = Int.MAX_VALUE,
    style: TextStyle = MaterialTheme.typography.bodyMedium
) {
    Text(
        text = parseMarkdown(text, color),
        color = color,
        style = style,
        maxLines = maxLines,
        overflow = TextOverflow.Ellipsis
    )
}

private fun parseMarkdown(input: String, textColor: Color): AnnotatedString {
    val builder = AnnotatedString.Builder()
    var i = 0
    while (i < input.length) {
        if (input.startsWith("**", i)) {
            val close = input.indexOf("**", i + 2)
            if (close > i) {
                builder.pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                builder.append(input.substring(i + 2, close))
                builder.pop()
                i = close + 2
                continue
            }
        }
        if (input[i] == '`') {
            val close = input.indexOf('`', i + 1)
            if (close > i) {
                builder.pushStyle(
                    SpanStyle(
                        fontFamily = FontFamily.Monospace,
                        background = textColor.copy(alpha = 0.08f)
                    )
                )
                builder.append(input.substring(i + 1, close))
                builder.pop()
                i = close + 1
                continue
            }
        }
        builder.append(input[i])
        i++
    }
    return builder.toAnnotatedString()
}
