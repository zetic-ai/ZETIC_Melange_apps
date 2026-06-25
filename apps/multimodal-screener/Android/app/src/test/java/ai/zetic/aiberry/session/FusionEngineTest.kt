package ai.zetic.aiberry.session

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Locks [FusionEngine] to iOS-Aiberry's `FusionEngine` math. Expected values are hand-computed
 * from the Swift formulas (canonical label order [Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise]):
 * face/voice blend by frame count, Russell valence/arousal projection, mid-band Energy/Rate,
 * composite 0.55*mood + 0.30*energyScore + 0.15*rateScore. If business logic changes, this fails.
 */
class FusionEngineTest {

    private fun oneHot(label: String): FloatArray {
        val labels = listOf("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
        return FloatArray(7) { if (labels[it] == label) 1f else 0f }
    }

    @Test
    fun happy_bothModalities_fullFrames() {
        // face==voice==Happy, 30 frames (face weight clamps to 0.8), voicedFraction 0.6.
        val r = FusionEngine.fuse(
            face = oneHot("Happy"), faceFrames = 30,
            voice = oneHot("Happy"), voicedFraction = 0.6f,
            transcript = emptyList(),
        )
        // valence 0.9 -> mood 95 ; arousal 0.6 -> energy 80 ; rate 100
        assertEquals(95, r.mood)
        assertEquals(80, r.energy)
        assertEquals(100, r.rateOfSpeech)
        // energyScore=midBand(80,55)=50 ; rateScore=midBand(100,60)=20
        // wellbeing = 0.55*95 + 0.30*50 + 0.15*20 = 70.25 -> 70
        assertEquals(70, r.wellbeing)
        assertEquals("Steady", r.band)
        assertEquals("Happy", r.faceTop?.label)
        assertEquals("Happy", r.voiceTop?.label)
        assertEquals("Happy", r.fused.first().label)
        assertEquals(listOf("Happy"), r.drivers["Mood"])
        assertEquals(1.0f, r.confidence, 1e-4f)
    }

    @Test
    fun sad_partialFrames_lowBand() {
        // face==voice==Sad, 15 frames (face weight 0.5), voicedFraction 0.3.
        val r = FusionEngine.fuse(
            face = oneHot("Sad"), faceFrames = 15,
            voice = oneHot("Sad"), voicedFraction = 0.3f,
            transcript = emptyList(),
        )
        // valence -0.8 -> mood 10 ; arousal -0.5 -> energy 25 ; rate 50
        assertEquals(10, r.mood)
        assertEquals(25, r.energy)
        assertEquals(50, r.rateOfSpeech)
        // energyScore=midBand(25,55)=40 ; rateScore=midBand(50,60)=80
        // wellbeing = 0.55*10 + 0.30*40 + 0.15*80 = 29.5 -> 29
        assertEquals(29, r.wellbeing)
        assertEquals("Low", r.band)
        assertEquals(0.75f, r.confidence, 1e-4f)
    }

    @Test
    fun voiceOnly_whenNoFace() {
        // No face data -> wFace 0, voice fully trusted; Neutral -> mid everything.
        val r = FusionEngine.fuse(
            face = FloatArray(0), faceFrames = 0,
            voice = oneHot("Neutral"), voicedFraction = 0.6f,
            transcript = emptyList(),
        )
        assertEquals(50, r.mood)        // valence 0 -> 50
        assertEquals(50, r.energy)      // arousal 0 -> 50
        assertEquals(100, r.rateOfSpeech)
        assertEquals("Neutral", r.voiceTop?.label)
        assertTrue(r.faceTop == null)
        assertEquals(0, r.faceFrames)
        assertEquals(1.0f, r.confidence, 1e-4f) // max(faceConf=0, voiceConf=1)
    }
}
