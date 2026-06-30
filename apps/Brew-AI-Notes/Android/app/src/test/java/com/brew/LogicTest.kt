package com.brew

import com.brew.asr.WhisperService
import com.brew.llm.LLMOutput
import com.brew.llm.Prompts
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests for the pure logic ports (no Android/native dependency). These
 * cover the highest-risk string/math behavior that would otherwise only surface
 * on a device.
 */
class LogicTest {

    // --- LLMOutput.sanitize (harmony/channel stripping) ---

    @Test fun sanitize_passesThroughPlainText() {
        assertEquals("Hello world", LLMOutput.sanitize("  Hello world  "))
    }

    @Test fun sanitize_dropsReasoningChannelKeepsFinal() {
        val raw = "<|channel|>analysis<|message|>thinking hard<|channel|>final<|message|>The answer is 42."
        val out = LLMOutput.sanitize(raw)
        assertTrue("kept final content", out.contains("The answer is 42."))
        assertFalse("dropped reasoning", out.contains("thinking hard"))
        assertFalse("stripped control tokens", out.contains("<|"))
    }

    @Test fun sanitize_stripsLeadingChannelLabel() {
        val out = LLMOutput.sanitize("<|message|>final: Done.")
        assertFalse(out.contains("<|"))
        assertTrue(out.contains("Done."))
    }

    // --- Prompts.truncateMiddle (keeps head + tail) ---

    @Test fun truncateMiddle_shortTextUnchanged() {
        val text = "short transcript"
        assertEquals(text, Prompts.truncateMiddle(text, 100))
    }

    @Test fun truncateMiddle_keepsHeadAndTail() {
        val head = "A".repeat(500)
        val tail = "Z".repeat(500)
        val text = head + "M".repeat(2000) + tail
        val out = Prompts.truncateMiddle(text, 200)
        assertTrue("has trim marker", out.contains("[middle of transcript trimmed]"))
        assertTrue("keeps head start", out.startsWith("A"))
        assertTrue("keeps tail end", out.endsWith("Z"))
        assertTrue("shorter than original", out.length < text.length)
    }

    @Test fun enhance_includesTranscriptAndInstruction() {
        val prompt = Prompts.enhance("We decided to ship on Friday.")
        assertTrue(prompt.contains("We decided to ship on Friday."))
        assertTrue(prompt.contains("Markdown"))
        assertTrue(prompt.contains("Transcript:"))
    }

    @Test fun enhance_handlesEmptyTranscript() {
        val prompt = Prompts.enhance("")
        assertTrue(prompt.contains("(no speech was transcribed)"))
    }

    // --- Whisper 30s windowing math ---

    @Test fun windowCount_isCeilingOf30sWindows() {
        assertEquals(0, WhisperService.windowCount(0))
        assertEquals(1, WhisperService.windowCount(1))
        assertEquals(1, WhisperService.windowCount(WhisperService.WINDOW_SAMPLES))
        assertEquals(2, WhisperService.windowCount(WhisperService.WINDOW_SAMPLES + 1))
        // 70s of audio -> 3 windows (30 + 30 + 10).
        assertEquals(3, WhisperService.windowCount(WhisperService.WINDOW_SAMPLES * 2 + 1))
    }

    @Test fun whisperForcedPromptTokensAreMultilingualOffsets() {
        // Guards against accidental edits to the forced English prompt ids.
        assertEquals(50258, WhisperService.SOT)
        assertEquals(50259, WhisperService.LANG_EN)
        assertEquals(50359, WhisperService.TRANSCRIBE)
        assertEquals(50363, WhisperService.NO_TIMESTAMPS)
        assertEquals(50257, WhisperService.EOT)
    }
}
