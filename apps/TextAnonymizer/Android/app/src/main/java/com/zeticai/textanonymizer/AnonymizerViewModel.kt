package com.zeticai.textanonymizer

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.tensor.DataType
import com.zeticai.mlange.core.tensor.Tensor
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

class AnonymizerViewModel(private val context: Context) : ViewModel() {
    companion object {
        private const val TAG = "TextAnonymizer"
    }
    
    private val _anonymizedText = MutableLiveData<String>()
    val anonymizedText: LiveData<String> = _anonymizedText
    
    private val _isProcessing = MutableLiveData<Boolean>()
    val isProcessing: LiveData<Boolean> = _isProcessing
    
    private val _showingError = MutableLiveData<Boolean>()
    val showingError: LiveData<Boolean> = _showingError
    
    private val _errorMessage = MutableLiveData<String>()
    val errorMessage: LiveData<String> = _errorMessage
    
    fun clearError() {
        _showingError.value = false
        _errorMessage.value = ""
    }
    
    private val _isModelLoaded = MutableLiveData<Boolean>()
    val isModelLoaded: LiveData<Boolean> = _isModelLoaded
    
    private var model: ZeticMLangeModel? = null
    private var tokenizer: Tokenizer? = null
    private val modelMaxLength = 128
    
    // Dynamic labels loaded from labels.json
    private var id2label: Map<Int, String> = emptyMap()
    
    private val placeholderByLabel = mapOf(
        "EMAIL" to "[Email]",
        "PHONE_NUMBER" to "[Phone number]",
        "CREDIT_CARD_NUMBER" to "[Credit card]",
        "SSN" to "[SSN]",
        "NRP" to "[NRP]",
        "PERSON" to "[Person]",
        "ADDRESS" to "[Address]",
        "LOCATION" to "[Location]",
        "DATE" to "[Date]",
        "OTHER" to "[Sensitive]"
    )
    
    init {
        loadModelAsync()
    }
    
    private fun loadModelAsync() {
        _isModelLoaded.value = false
        
        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.i(TAG, "Starting model loading...")
                
                // 1. Load Tokenizer
                tokenizer = Tokenizer(context)
                
                // 2. Load Labels
                loadLabels()
                
                if (Constants.PERSONAL_KEY.isEmpty() || 
                    Constants.PERSONAL_KEY == "YOUR_MLANGE_KEY") {
                    throw Exception("Missing personal key. Set PERSONAL_KEY in Constants.kt")
                }
                
                val loadedModel = ZeticMLangeModel(
                    context.applicationContext,
                    Constants.PERSONAL_KEY,
                    Constants.MODEL_ID
                )
                
                Log.i(TAG, "Model loaded successfully")
                
                withContext(Dispatchers.Main) {
                    model = loadedModel
                    _isModelLoaded.value = true
                }
            } catch (e: Exception) {
                val errorDescription = e.message ?: e.toString()
                Log.e(TAG, "Model loading failed: $errorDescription")
                
                withContext(Dispatchers.Main) {
                    _isModelLoaded.value = false
                    _errorMessage.value = "Failed to load: $errorDescription"
                    _showingError.value = true
                }
            }
        }
    }

    private fun loadLabels() {
        try {
            val inputStream = context.assets.open("labels.json")
            val content = inputStream.bufferedReader().use { it.readText() }
            val json = JSONObject(content)
            val map = HashMap<Int, String>()
            val keys = json.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                val value = json.getString(key)
                map[key.toInt()] = value
            }
            id2label = map
            Log.d(TAG, "Loaded ${id2label.size} labels.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load labels.json", e)
            throw e
        }
    }
    
    fun anonymizeText(text: String) {
        if (text.isEmpty()) return
        
        val currentModel = model
        val currentTokenizer = tokenizer
        
        if (currentModel == null || currentTokenizer == null) {
            _errorMessage.value = "Model or Tokenizer not ready."
            _showingError.value = true
            return
        }
        
        _isProcessing.value = true
        _anonymizedText.value = ""
        
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // 1. Tokenize
                val (inputIds, attentionMask) = tokenize(text, currentTokenizer)
                
                // 2. Run Inference
                val finalInputIds = if (inputIds.size > modelMaxLength) inputIds.take(modelMaxLength).toLongArray() else inputIds + LongArray(modelMaxLength - inputIds.size) { currentTokenizer.padId.toLong() }
                val finalMask = if (attentionMask.size > modelMaxLength) attentionMask.take(modelMaxLength).toLongArray() else attentionMask + LongArray(modelMaxLength - attentionMask.size) { 0 }
                
                val inputs = arrayOf(
                    createLongTensor(finalInputIds, intArrayOf(1, modelMaxLength), "input_ids"),
                    createLongTensor(finalMask, intArrayOf(1, modelMaxLength), "attention_mask")
                )
                
                val outputs = currentModel.run(inputs)
                
                // 3. Post-process (BIO Decoding)
                if (outputs.isNotEmpty()) {
                    val result = decodeAndAnonymize(outputs[0], finalInputIds, finalMask, currentTokenizer)
                    withContext(Dispatchers.Main) {
                        _anonymizedText.value = result
                        _isProcessing.value = false
                    }
                } else {
                     throw Exception("No output from model")
                }
                
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    _isProcessing.value = false
                    _errorMessage.value = "Error: ${e.message}"
                    _showingError.value = true
                }
            }
        }
    }
    
    private fun tokenize(text: String, tokenizer: Tokenizer): Pair<LongArray, LongArray> {
        val ids = tokenizer.encode(text)
        val mask = LongArray(ids.size) { 1 }
        return Pair(ids, mask)
    }

    private fun createLongTensor(values: LongArray, shape: IntArray, label: String): Tensor {
        return Tensor.Companion.of(values, DataType.Companion.from("int64"), shape, false)
    }

    private fun decodeAndAnonymize(logitsTensor: Tensor, inputIds: LongArray, attentionMask: LongArray, tokenizer: Tokenizer): String {
        val floats = tensorToFloatArray(logitsTensor) ?: return "Error parsing output"
        val classCount = id2label.size
        
        // Safety check
        if (classCount == 0) return "Labels not loaded"
        
        val seqLen = floats.size / classCount
        
        // 1. Get predictions (argmax)
        val predIds = IntArray(seqLen)
        for (i in 0 until seqLen) {
            var maxScore = Float.NEGATIVE_INFINITY
            var maxIdx = 0
            val offset = i * classCount
            for (c in 0 until classCount) {
                if (offset + c < floats.size) {
                    val score = floats[offset + c]
                    if (score > maxScore) {
                        maxScore = score
                        maxIdx = c
                    }
                }
            }
            predIds[i] = maxIdx
        }

        // 2. Reconstruct masked labels
        val maskedTokens = ArrayList<String>()
        
        var i = 0
        // Use actual sequence length (ignoring global padding, but respecting inputIds length)
        val realLen = minOf(seqLen, inputIds.size) 
        
        while (i < realLen) {
             // Skip special tokens in output if needed, but logic usually iterates all
             // Attention mask check
             if (i < attentionMask.size && attentionMask[i] == 0L) {
                 i++
                 continue
             }
             
             // Check if it's a special token (CLS, SEP, PAD)
             val currentId = inputIds[i].toInt()
             if (currentId == tokenizer.bosId || currentId == tokenizer.eosId || currentId == tokenizer.padId) {
                 i++
                 continue
             }
             
             val label = id2label[predIds[i]] ?: "O"
             val rawToken = tokenizer.getRawToken(currentId) ?: ""
             
             if (label == "O") {
                 maskedTokens.add(tokenizer.decodeToken(currentId))
                 i++
                 continue
             }
             
             // B-Entity or I-Entity
             var entityType = label
             if (label.startsWith("B-") || label.startsWith("I-")) {
                 entityType = label.substring(2)
             }
             
             // Python logic:
             // maskedTokens.append("[MASKED]") --> mapped to placeholder
             var placeholder = placeholderByLabel[entityType] ?: "[$entityType]"
             
             // Preserve leading space if any
             if (rawToken.startsWith("\u0120")) {
                 placeholder = "\u0120" + placeholder
             }
             
             maskedTokens.add(placeholder)
             
             i++
             // Skip consecutive I- of same type
             while (i < realLen) {
                 if (i < attentionMask.size && attentionMask[i] == 0L) break
                 
                 val nextId = inputIds[i].toInt()
                  if (nextId == tokenizer.eosId || nextId == tokenizer.padId) break
                 
                 val nextLabel = id2label[predIds[i]] ?: "O"
                 if (nextLabel == "I-$entityType" || nextLabel == "B-$entityType") {
                     // In python: if next_label == "I-" or "B-" (some models predict B repeatedly)
                     // skip this token as it's part of the same entity already masked
                     i++
                 } else {
                     break
                 }
             }
        }
        
        // 3. Join tokens
        // Replace unicode space placeholder if any
        val result = maskedTokens.joinToString("")
            .replace("\u0120", " ")
        
        return result.trim()
    }
    
    private fun tensorToFloatArray(tensor: Tensor): FloatArray? {
        val buffer = tensor.data
        val floats = FloatArray(buffer.remaining() / 4)
        buffer.asFloatBuffer().get(floats)
        return floats
    }
}
