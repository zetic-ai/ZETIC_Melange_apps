package ai.zetic.demo.cherrypad

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import ai.zetic.demo.cherrypad.ui.RootScreen
import ai.zetic.demo.cherrypad.ui.theme.CherryTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            CherryTheme {
                RootScreen()
            }
        }
    }
}
