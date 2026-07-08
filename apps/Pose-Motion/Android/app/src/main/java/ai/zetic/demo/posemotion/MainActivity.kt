package ai.zetic.demo.posemotion

import ai.zetic.demo.posemotion.state.DemoViewModel
import ai.zetic.demo.posemotion.ui.PoseMotionTheme
import ai.zetic.demo.posemotion.ui.RootScreen
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels

class MainActivity : ComponentActivity() {
    private val viewModel: DemoViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PoseMotionTheme {
                RootScreen(viewModel)
            }
        }
    }
}
