package com.zeticai.textanonymizer

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.doAfterTextChanged
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.zeticai.textanonymizer.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private val viewModel: AnonymizerViewModel by lazy {
        ViewModelProvider(
            this,
            viewModelFactory {
                initializer {
                    AnonymizerViewModel(this@MainActivity)
                }
            }
        )[AnonymizerViewModel::class.java]
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setupObservers()
        setupClickListeners()
        setupTextWatcher()
        updateAnonymizeButtonState()
        setupExampleButtons()
    }
    
    private fun setupObservers() {
        // Observe model loading state
        viewModel.isModelLoaded.observe(this) { isLoaded ->
            if (isLoaded) {
                binding.progressModelLoading.visibility = View.GONE
                binding.textModelLoading.visibility = View.GONE
            } else {
                binding.progressModelLoading.visibility = View.VISIBLE
                binding.textModelLoading.visibility = View.VISIBLE
            }
            updateAnonymizeButtonState()
        }
        
        // Observe processing state
        viewModel.isProcessing.observe(this) { isProcessing ->
            updateAnonymizeButtonState()
            if (isProcessing) {
                binding.progressAnonymizing.visibility = View.VISIBLE
                binding.buttonAnonymize.text = "Anonymizing..."
            } else {
                binding.progressAnonymizing.visibility = View.GONE
                binding.buttonAnonymize.text = "Anonymize Text"
            }
        }
        
        // Observe anonymized text
        viewModel.anonymizedText.observe(this) { text ->
            if (text.isNotEmpty()) {
                binding.textAnonymizedOutput.text = text
                binding.layoutOutputSection.visibility = View.VISIBLE
            } else {
                binding.layoutOutputSection.visibility = View.GONE
            }
        }
        
        // Observe errors
        viewModel.showingError.observe(this) { showing ->
            if (showing) {
                val message = viewModel.errorMessage.value ?: "An error occurred"
                AlertDialog.Builder(this)
                    .setTitle("Error")
                    .setMessage(message)
                    .setPositiveButton("OK") { _, _ ->
                        // Error dialog dismissed
                    }
                    .setOnDismissListener {
                        // Reset error state after dialog is dismissed
                        viewModel.clearError()
                    }
                    .show()
            }
        }
    }
    
    private fun setupClickListeners() {
        // Anonymize button
        binding.buttonAnonymize.setOnClickListener {
            val inputText = binding.editTextInput.text.toString()
            if (inputText.isNotEmpty()) {
                hideKeyboard()
                viewModel.anonymizeText(inputText)
            }
        }
        
        // Copy button
        binding.buttonCopy.setOnClickListener {
            val text = viewModel.anonymizedText.value
            if (!text.isNullOrEmpty()) {
                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                val clip = ClipData.newPlainText("Anonymized Text", text)
                clipboard.setPrimaryClip(clip)
                Toast.makeText(this, "Copied to clipboard", Toast.LENGTH_SHORT).show()
            }
        }
        
        // Share button
        binding.buttonShare.setOnClickListener {
            val text = viewModel.anonymizedText.value
            if (!text.isNullOrEmpty()) {
                val shareIntent = Intent().apply {
                    action = Intent.ACTION_SEND
                    putExtra(Intent.EXTRA_TEXT, text)
                    type = "text/plain"
                }
                startActivity(Intent.createChooser(shareIntent, "Share anonymized text"))
            }
        }
    }

    private fun setupTextWatcher() {
        binding.editTextInput.doAfterTextChanged {
            updateAnonymizeButtonState()
        }
    }
    
    private fun updateAnonymizeButtonState() {
        val inputText = binding.editTextInput.text.toString()
        val isModelLoaded = viewModel.isModelLoaded.value ?: false
        val isProcessing = viewModel.isProcessing.value ?: false
        
        val isEnabled = (inputText.isNotEmpty() && isModelLoaded && !isProcessing)
        binding.buttonAnonymize.isEnabled = isEnabled
        
        // Update button appearance
        binding.buttonAnonymize.alpha = if (isEnabled) 1.0f else 0.5f
        
        // Update button text if not processing
        if (!isProcessing) {
            binding.buttonAnonymize.text = "Anonymize Text"
        }
    }
    
    private fun hideKeyboard() {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(binding.editTextInput.windowToken, 0)
    }

    private fun setupExampleButtons() {
        val examples = mapOf(
            R.id.btnCase1 to "Mr. Sherlock Holmes received a visitor at 221B Baker Street, London on October 31st, 1890. The client, Jabez Wilson, had a strange story about a Red-Headed League which dissolved unexpectedly.",
            R.id.btnCase2 to "Tony Stark held a press conference in New York on May 2, 2008. He declared 'I am Iron Man' to the reporters, causing Stark Industries stock to rise by 15% immediately after the announcement.",
            R.id.btnCase3 to "Harry Potter received his first letter at 4 Privet Drive, Little Whinging, Surrey on his 11th birthday, July 31, 1991. Rubeus Hagrid later arrived to personally deliver the invitation to Hogwarts.",
            R.id.btnCase4 to "Jack Dawson won his ticket to the Titanic in a highly lucky poker game on April 10, 1912. The ship departed from Southampton shortly after and was scheduled to arrive in New York City.",
            R.id.btnCase5 to "Project Manager Sarah Connor called 555-0199 to schedule an urgent meeting with Miles Dyson. They agreed to meet at 123 Cyberdyne Systems Way, California on August 29th to discuss the neural net processor.",
            R.id.btnCase6 to "Patient John Doe (ID: 998-877-66) was admitted to Princeton-Plainsboro Teaching Hospital on November 21st. Dr. Gregory House ordered an MRI and a lumbar puncture immediately despite the team's initial hesitation.",
            R.id.btnCase7 to "On June 17, 1994, a white Ford Bronco was driven by Al Cowlings in Los Angeles. The passenger, O.J. Simpson, was subsequently charged, and the trial began on January 24, 1995.",
            R.id.btnCase8 to "Please transfer 1,000,000 USD to the account 1234-5678-9012 held by Walter White. The transaction must be completed by December 25th at the Albuquerque branch to avoid suspicion from the DEA.",
            R.id.btnCase9 to "Order ID 556677 for Frodo Baggins cannot be delivered to Bag End, Hobbiton due to the recipient's absence. Please reroute the package to the Prancing Pony Inn, Bree, by September 22nd.",
            R.id.btnCase10 to "On July 20, 1969, Neil Armstrong and Buzz Aldrin landed the Eagle on the Moon surface. They planted the US flag and returned to Earth, splashing down in the Pacific Ocean on July 24th."
        )

        examples.forEach { (id, text) ->
            findViewById<Button>(id)?.setOnClickListener {
                binding.editTextInput.setText(text)
            }
        }
    }
}
