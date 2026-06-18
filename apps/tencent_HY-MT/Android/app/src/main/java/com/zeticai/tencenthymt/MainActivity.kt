package com.zeticai.tencenthymt

import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.zeticai.mlange.core.model.llm.LLMModelMode
import com.zeticai.mlange.core.model.llm.ZeticMLangeLLMModel
import com.zeticai.tencenthymt.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    object Constants {
        const val MLANGE_PERSONAL_ACCESS_TOKEN = "YOUR_MLANGE_KEY"
        const val MODEL_NAME = "vaibhav-zetic/tencent_HY-MT"
    }

    private var activeModel: ZeticMLangeLLMModel? = null
    private var currentSourceLang: Language? = null
    private var currentTargetLang: Language? = null
    
    // Track if spinners have been initialized to avoid redundant loads on startup
    private var isSpinnersInitialized = false
    private var isSwapping = false

    @Synchronized
    private fun getOrUpdateModel(source: Language, target: Language, onProgress: ((Float) -> Unit)? = null): ZeticMLangeLLMModel {
        if (activeModel == null) {
            currentSourceLang = source
            currentTargetLang = target
            activeModel = createModel(onProgress ?: {})
        } else {
            val isSwap = source == currentTargetLang && target == currentSourceLang
            val isChange = source != currentSourceLang || target != currentTargetLang

            if (isChange) {
                currentSourceLang = source
                currentTargetLang = target
                
                // Always clean up context when language direction changes to avoid model confusion
                activeModel?.cleanUp()
            }
        }
        return activeModel!!
    }

    private fun createModel(onProgress: (Float) -> Unit = {}): ZeticMLangeLLMModel {
        return ZeticMLangeLLMModel(
            this,
            Constants.MLANGE_PERSONAL_ACCESS_TOKEN,
            Constants.MODEL_NAME,
            null,
            modelMode = LLMModelMode.RUN_AUTO,
            onProgress = onProgress
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.sendButton.isEnabled = false // Disable until loaded
        binding.statusText.text = "Initializing Model..."

        // Trigger initial model loading (download if needed)
        val defaultSource = Languages.list.first { it.name == "English" }
        val defaultTarget = Languages.list.first { it.name == "Korean" }

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Initialize model with defaults
                getOrUpdateModel(defaultSource, defaultTarget) { progress ->
                    runOnUiThread {
                        binding.messageInput.hint = "File Downloading... ${(progress * 100).toInt()}%"
                    }
                }

                // Model is ready (either because it was cached or just downloaded)
                runOnUiThread {
                    binding.messageInput.hint = "Type a message..."
                    binding.statusText.text = "Model Loaded"
                    binding.sendButton.isEnabled = true
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    binding.statusText.text = "Error initializing: ${e.toString()}"
                }
            }
        }

        // Setup Spinners
        val adapter = android.widget.ArrayAdapter(
            this,
            R.layout.spinner_item,
            Languages.list
        )
        binding.sourceLangSpinner.adapter = adapter
        binding.targetLangSpinner.adapter = adapter

        // set default: English -> Korean
        binding.sourceLangSpinner.setSelection(Languages.list.indexOfFirst { it.name == "English" })
        binding.targetLangSpinner.setSelection(Languages.list.indexOfFirst { it.name == "Korean" })
        
        // Listener to eagerly load model on language change
        val spinnerListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!isSpinnersInitialized || isSwapping) {
                    isSpinnersInitialized = true
                    return
                }
                
                val source = binding.sourceLangSpinner.selectedItem as Language
                val target = binding.targetLangSpinner.selectedItem as Language
                
                // Pre-load/Switch model in background immediately
                lifecycleScope.launch(Dispatchers.IO) {
                    try {
                        getOrUpdateModel(source, target) { progress ->
                            runOnUiThread {
                                binding.statusText.text = "Switching Model... ${(progress * 100).toInt()}%"
                            }
                        }
                            runOnUiThread { 
                                 if (binding.statusText.text.startsWith("Switching")) { 
                                     binding.statusText.text = "Model Ready" 
                                     binding.sendButton.isEnabled = true
                                 }
                            }
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        
        binding.sourceLangSpinner.onItemSelectedListener = spinnerListener
        binding.targetLangSpinner.onItemSelectedListener = spinnerListener

        binding.swapButton.setOnClickListener {
            isSwapping = true // Block listeners
            val sourcePos = binding.sourceLangSpinner.selectedItemPosition
            val targetPos = binding.targetLangSpinner.selectedItemPosition
            binding.sourceLangSpinner.setSelection(targetPos)
            binding.targetLangSpinner.setSelection(sourcePos)
            isSwapping = false // Unblock
            
            // Manual update with final state using explicit indices
            // NOTE: spinner.selectedItem might not be updated yet if checked immediately after setSelection
            val source = Languages.list[targetPos]
            val target = Languages.list[sourcePos]
            
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    getOrUpdateModel(source, target)
                    runOnUiThread {
                        // Don't just set text "Model Ready", check real status or leave it to getOrUpdateModel side effects if any?
                        // Actually getOrUpdateModel doesn't emit "Model Ready" if it just returns activeModel.
                        // So we should enforce it here.
                        binding.statusText.text = "Model Ready"
                        binding.sendButton.isEnabled = true
                    }
                } catch(e: Exception) {
                    e.printStackTrace()
                }
            }
        }

        binding.sendButton.setOnClickListener {
            val inputText = binding.messageInput.text.toString()
            if (inputText.isNotEmpty()) {
                val sourceLang = binding.sourceLangSpinner.selectedItem as Language
                val targetLang = binding.targetLangSpinner.selectedItem as Language

                // Add User Bubble
                addMessageBubble(inputText, true)
                binding.messageInput.text.clear()

                // Construct Prompt
                val prompt = "Translate the following segment from ${sourceLang.name} into ${targetLang.name}, without additional explanation. \n$inputText"


                // Log debug info
                Log.d("TencentHYMT", "Translating from ${sourceLang.name} into ${targetLang.name} with prompt: \n$prompt")

                // Add Model Placeholder Bubble
                val modelBubble = addMessageBubble("Generating...", false)

                // Perform inference in background
                Thread {
                    var model: ZeticMLangeLLMModel? = null
                    try {
                        // Get or Update model (reuses if languages match)
                        model = getOrUpdateModel(sourceLang, targetLang) {
                             // Progress callback for switch case if download needed (unlikely if already cached)
                             runOnUiThread { binding.statusText.text = "Loading Model..." }
                        }
                        
                        model.run(prompt)
                        var isFirstToken = true
                        while (true) {
                            val token = model.waitForNextToken()
                            if (token == "") break

                            runOnUiThread {
                                if (isFirstToken) {
                                    modelBubble.text = "" // Clear placeholder
                                    isFirstToken = false
                                }
                                modelBubble.append(token)
                                binding.chatScroll.post {
                                    binding.chatScroll.fullScroll(View.FOCUS_DOWN)
                                }
                            }
                        }
                    } catch (e: Exception) {
                        runOnUiThread {
                            modelBubble.text = "Error: ${e.message}"
                        }
                    } finally {
                        try {
                            model?.cleanUp()
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }
                }.start()
            }
        }
    }

    private fun addMessageBubble(text: String, isUser: Boolean): TextView {
        val textView = TextView(this)
        textView.text = text
        textView.textSize = 16f
        val density = resources.displayMetrics.density
        val padding = (12 * density).toInt()
        textView.setPadding(padding, padding, padding, padding)

        val params = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        )

        if (isUser) {
            params.gravity = Gravity.END
            textView.background = ContextCompat.getDrawable(this, R.drawable.bg_user_bubble)
            textView.setTextColor(ContextCompat.getColor(this, R.color.text_bubble_user))
            params.marginStart = (48 * density).toInt() // Indent from left
        } else {
            params.gravity = Gravity.START
            textView.background = ContextCompat.getDrawable(this, R.drawable.bg_model_bubble)
            textView.setTextColor(ContextCompat.getColor(this, R.color.text_bubble_model))
            params.marginEnd = (48 * density).toInt() // Indent from right
        }
        params.bottomMargin = (8 * density).toInt()

        textView.layoutParams = params
        binding.chatContainer.addView(textView)

        // Auto scroll
        binding.chatScroll.post {
            binding.chatScroll.fullScroll(View.FOCUS_DOWN)
        }

        textView.setOnLongClickListener {
            val clipboard = getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
            val clip = android.content.ClipData.newPlainText("Copied Text", textView.text)
            clipboard.setPrimaryClip(clip)
            android.widget.Toast.makeText(this, "Text Copied", android.widget.Toast.LENGTH_SHORT).show()
            true
        }

        return textView
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            activeModel?.cleanUp()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
