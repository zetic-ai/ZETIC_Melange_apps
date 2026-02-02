package com.zeticai.textanonymizer

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

class Tokenizer(context: Context) {
    private val vocab = HashMap<String, Int>()
    private val idToToken = HashMap<Int, String>()
    
    // Default special tokens for RoBERTa/BERT style models
    var bosId = 0   // [CLS] or <s>
    var eosId = 2   // [SEP] or </s>
    var unkId = 3   // [UNK] or <unk>
    var padId = 1   // [PAD] or <pad>
    var maskId = 4

    init {
        loadVocab(context)
    }

    private fun loadVocab(context: Context) {
        try {
            val inputStream = context.assets.open("tokenizer.json")
            val reader = BufferedReader(InputStreamReader(inputStream))
            val sb = StringBuilder()
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                sb.append(line)
            }
            reader.close()

            val root = JSONObject(sb.toString())
            
            // Allow for different tokenizer.json structures
            // Usually found in model -> vocab
            if (root.has("model")) {
                val model = root.getJSONObject("model")
                if (model.has("vocab")) {
                    val vocabObj = model.getJSONObject("vocab")
                    val keys = vocabObj.keys()
                    while (keys.hasNext()) {
                        val token = keys.next()
                        val id = vocabObj.getInt(token)
                        vocab[token] = id
                        idToToken[id] = token
                    }
                }
            } else if (root.has("vocab")) {
                 val vocabObj = root.getJSONObject("vocab")
                 val keys = vocabObj.keys()
                 while (keys.hasNext()) {
                     val token = keys.next()
                     val id = vocabObj.getInt(token)
                     vocab[token] = id
                     idToToken[id] = token
                 }
            }
            
            // Try to find helper tokens
            bosId = vocab["<s>"] ?: vocab["[CLS]"] ?: bosId
            eosId = vocab["</s>"] ?: vocab["[SEP]"] ?: eosId
            unkId = vocab["<unk>"] ?: vocab["[UNK]"] ?: unkId
            padId = vocab["<pad>"] ?: vocab["[PAD]"] ?: padId
            maskId = vocab["<mask>"] ?: vocab["[MASK]"] ?: maskId

            Log.d("Tokenizer", "Loaded ${vocab.size} tokens.")
        } catch (e: Exception) {
            Log.e("Tokenizer", "Failed to load vocab", e)
        }
    }

    // Basic BPE Encoding Logic for RoBERTa/GPT-2 style tokenizers
    // 1. Split by whitespace
    // 2. Add prefix space if not start of string (RoBERTa convention: "Ġ")
    // 3. Greedy match against vocab
    fun encode(text: String): LongArray {
        val ids = ArrayList<Long>()
        ids.add(bosId.toLong())

        // In RoBERTa/XLM-R, spaces are often represented as \u0120 (Ġ)
        // We do a simplified pass: split by space, prepend Ġ to all except maybe first if it doesn't have space?
        // Actually, HuggingFace tokenizer usually treats the input string as a stream.
        // Rule: Replace space with Ġ (\u0120)
        // But simply replacing space with Ġ isn't enough for BPE split.
        
        // Simplified Logic:
        // 1. Prepend space to text (common trick for RoBERTa to handle start of sentence)
        // 2. Replace ' ' with 'Ġ'
        // 3. For each character sequence, try to find longest match in vocab
        
        // Note: Real implementations use a priority queue merge (BPE marks).
        // Since we don't have the "merges" list from tokenizer.json easily accessible/parsed here without complexity,
        // we will fall back to a "Greedy WordPiece/Unigram" style approach which works "okay" for many models
        // provided the vocab has the full words.
        
        // HOWEVER: RoBERTa is BPE. If we just greedy match "Apple", and "Apple" isn't in vocab
        // but "Ap" and "ple" are, we need to find that.
        
        // Let's implement a standard greedy max-match against vocab.
        // It's not 100% correct BPE (which is frequency based), but close enough for anonymization triggers.
        
        val processedText = " " + text // Prepend space
        val cleanText = processedText.replace(" ", "\u0120") // Replace space with Ġ
        
        var i = 0
        while (i < cleanText.length) {
            var matchFound = false
            // Greedy match
            val maxSearchLen = minOf(cleanText.length - i, 20) // Limit token length to avoid O(N^2) too deep
            
            for (len in maxSearchLen downTo 1) {
                val sub = cleanText.substring(i, i + len)
                if (vocab.containsKey(sub)) {
                    ids.add(vocab[sub]!!.toLong())
                    i += len
                    matchFound = true
                    break
                }
            }
            
            if (!matchFound) {
                // Try char by char as UNK or bytes
                // For now, map to UNK to avoid infinite loop
                ids.add(unkId.toLong())
                i++
            }
        }

        ids.add(eosId.toLong())
        return ids.toLongArray()
    }

    fun decode(tokens: LongArray): String {
         val sb = StringBuilder()
         for (t in tokens) {
             if (t == eosId.toLong() || t == padId.toLong()) continue
             if (t == bosId.toLong()) continue
             
             var s = idToToken[t.toInt()] ?: ""
             s = s.replace("\u0120", " ")
             sb.append(s)
         }
         return sb.toString().trim()
    }
    
    fun decodeToken(tokenId: Int): String {
        var s = idToToken[tokenId] ?: ""
        return s.replace("\u0120", " ")
    }
    
    fun getRawToken(tokenId: Int): String? {
        return idToToken[tokenId]
    }
}
