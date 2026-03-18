package com.zeticai.qwen3chat.llm

import android.content.Context
import com.zeticai.mlange.core.model.llm.ZeticMLangeLLMModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive

class LLMService(private val context: Context) {
    private var model: ZeticMLangeLLMModel? = null
    
    val modelId = "Qwen/Qwen3-4B"
    // PERSONAL KEY is hidden from logs and directly used
    private val personalKey = "YOUR_MLANGE_KEY"

    fun initialize(onDownloadProgress: (Float) -> Unit) {
        if (model == null) {
            model = ZeticMLangeLLMModel(context, personalKey, modelId) { progress ->
                onDownloadProgress(progress)
            }
        }
    }

    suspend fun generateResponse(prompt: String): Flow<TokenSync> = flow {
        val llm = model ?: throw IllegalStateException("Model not initialized")
        
        var totalTokens = 0
        val startTime = System.currentTimeMillis()
        
        // Safety clear before run as per Cleanup Contract
        llm.cleanUp() 
        llm.run(prompt)
        
        while (currentCoroutineContext().isActive) {
            val result = llm.waitForNextToken()
            if (result.generatedTokens == 0) break
            totalTokens++
            emit(TokenSync.Token(result.token, totalTokens))
        }
        val duration = System.currentTimeMillis() - startTime
        emit(TokenSync.Done(totalTokens, duration))
        
        // Cleanup when done
        llm.cleanUp()
    }.flowOn(Dispatchers.IO)
    
    fun stop() {
        model?.cleanUp()
    }
    
    fun clear() {
        model?.cleanUp()
    }
}

sealed class TokenSync {
    data class Token(val text: String, val count: Int) : TokenSync()
    data class Done(val totalTokens: Int, val durationMs: Long) : TokenSync()
}
