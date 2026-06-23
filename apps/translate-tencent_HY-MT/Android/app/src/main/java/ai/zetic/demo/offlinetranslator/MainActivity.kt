package ai.zetic.demo.offlinetranslator

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import ai.zetic.demo.offlinetranslator.theme.Theme
import ai.zetic.demo.offlinetranslator.theme.OfflineTranslatorTheme
import ai.zetic.demo.offlinetranslator.ui.TranslatorScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            OfflineTranslatorTheme {
                TranslatorScreen(Modifier.fillMaxSize().background(Theme.background))
            }
        }
    }
}
