package com.brew.asr

import android.content.Context
import org.json.JSONObject

/**
 * Pure-Kotlin GPT-2 byte-level BPE detokenizer for Whisper token ids, ported
 * from VoxScribe's `detokenizer.dart`.
 *
 *   id -> token unicode string (via bundled `vocab.json`, token->id)
 *      -> map each char back to a raw byte (reverse of GPT-2 bytes_to_unicode)
 *      -> UTF-8 decode the assembled byte stream.
 *
 * Special / non-vocab ids (EOT, pad, SOT, language/task/timestamp tokens) have
 * no `vocab.json` entry and are skipped.
 */
class WhisperDetokenizer private constructor(
    private val idToToken: Array<String?>,
    private val byteDecoder: Map<Int, Int>,
) {
    /** Decodes token ids to text, skipping specials. */
    fun decode(ids: IntArray): String {
        val bytes = ArrayList<Byte>(ids.size * 2)
        for (id in ids) {
            if (id >= EOT || id == PAD || id == SOT) continue
            if (id < 0 || id >= idToToken.size) continue
            val token = idToToken[id] ?: continue
            var i = 0
            while (i < token.length) {
                val cp = token.codePointAt(i)
                i += Character.charCount(cp)
                val b = byteDecoder[cp]
                if (b != null) bytes.add(b.toByte())
            }
        }
        if (bytes.isEmpty()) return ""
        return String(bytes.toByteArray(), Charsets.UTF_8)
    }

    companion object {
        const val EOT = 50257
        const val PAD = 50256
        const val SOT = 50258

        fun fromAsset(context: Context, assetName: String = "vocab.json"): WhisperDetokenizer {
            val json = context.assets.open(assetName).use { it.readBytes().toString(Charsets.UTF_8) }
            val obj = JSONObject(json)
            var maxId = 0
            val keys = obj.keys()
            val entries = ArrayList<Pair<String, Int>>(obj.length())
            while (keys.hasNext()) {
                val token = keys.next()
                val id = obj.getInt(token)
                entries.add(token to id)
                if (id > maxId) maxId = id
            }
            val idToToken = arrayOfNulls<String>(maxId + 1)
            for ((token, id) in entries) idToToken[id] = token
            return WhisperDetokenizer(idToToken, buildByteDecoder())
        }

        /** GPT-2 bytes_to_unicode, reversed (unicode code point -> raw byte). */
        private fun buildByteDecoder(): Map<Int, Int> {
            val bs = ArrayList<Int>()
            for (i in '!'.code..'~'.code) bs.add(i)
            for (i in 0xA1..0xAC) bs.add(i)
            for (i in 0xAE..0xFF) bs.add(i)
            val cs = ArrayList(bs)
            var n = 0
            for (b in 0 until 256) {
                if (!bs.contains(b)) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            val decoder = HashMap<Int, Int>()
            for (i in bs.indices) decoder[cs[i]] = bs[i]
            return decoder
        }
    }
}
