package com.brew.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.brew.ui.theme.BrewColors
import com.brew.ui.theme.Serif

/**
 * Lightweight Markdown renderer (ported from iOS `MarkdownView`): line-based,
 * `#`/`##` headings, bullets, and inline bold/italic.
 */
@Composable
fun MarkdownText(markdown: String, modifier: Modifier = Modifier) {
    val lines = markdown.split("\n")
    Column(modifier = modifier) {
        for (raw in lines) {
            val line = raw.trimEnd()
            val trimmed = line.trimStart()
            when {
                trimmed.startsWith("# ") -> Text(
                    text = inline(trimmed.removePrefix("# ")),
                    fontFamily = Serif,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color = BrewColors.ink,
                    modifier = Modifier.padding(top = 10.dp, bottom = 2.dp),
                )

                trimmed.startsWith("## ") -> Text(
                    text = inline(trimmed.removePrefix("## ")),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = BrewColors.ink,
                    modifier = Modifier.padding(top = 6.dp, bottom = 2.dp),
                )

                trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("• ") ->
                    Row(modifier = Modifier.padding(vertical = 3.dp)) {
                        Text("•", fontSize = 17.sp, color = BrewColors.inkSecondary)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = inline(trimmed.drop(2)),
                            fontSize = 17.sp,
                            color = BrewColors.ink,
                        )
                    }

                trimmed.isEmpty() -> Spacer(Modifier.height(6.dp))

                else -> Text(
                    text = inline(trimmed),
                    fontSize = 17.sp,
                    color = BrewColors.ink,
                    modifier = Modifier.padding(vertical = 2.dp),
                )
            }
        }
    }
}

/** Parses inline `**bold**`, `*italic*`/`_italic_` into an AnnotatedString. */
private fun inline(text: String): AnnotatedString = buildAnnotatedString {
    var i = 0
    while (i < text.length) {
        val c = text[i]
        if (c == '*' && i + 1 < text.length && text[i + 1] == '*') {
            val end = text.indexOf("**", i + 2)
            if (end > i + 1) {
                withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                    append(text.substring(i + 2, end))
                }
                i = end + 2
                continue
            }
        }
        if (c == '*' || c == '_') {
            val end = text.indexOf(c, i + 1)
            if (end > i) {
                withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                    append(text.substring(i + 1, end))
                }
                i = end + 1
                continue
            }
        }
        append(c)
        i++
    }
}
