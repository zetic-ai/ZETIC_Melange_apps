package ai.zetic.demo.cameravitals

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import ai.zetic.demo.cameravitals.ui.CameraVitalsTheme
import ai.zetic.demo.cameravitals.ui.RootScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CameraVitalsTheme {
                RootScreen()
            }
        }
    }
}
