package ai.zetic.voicevitals

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Air
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.sp
import ai.zetic.voicevitals.core.AppViewModel
import ai.zetic.voicevitals.emotion.EmotionScreen
import ai.zetic.voicevitals.respiratory.RespiratoryScreen
import ai.zetic.voicevitals.ui.AboutScreen
import ai.zetic.voicevitals.ui.Theme

class MainActivity : ComponentActivity() {
    private val appViewModel: AppViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            LaunchedEffect(Unit) { appViewModel.preloadAll() }
            VoiceVitalsApp(appViewModel)
        }
    }
}

private data class Tab(val label: String, val icon: ImageVector)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun VoiceVitalsApp(vm: AppViewModel) {
    var selected by remember { mutableIntStateOf(0) }
    val tabs = remember {
        listOf(
            Tab("Emotion", Icons.Filled.GraphicEq),
            Tab("Respiratory", Icons.Filled.Air),
            Tab("About", Icons.Filled.Info),
        )
    }

    Scaffold(
        containerColor = Theme.Bg,
        bottomBar = {
            NavigationBar(containerColor = Theme.Bg) {
                tabs.forEachIndexed { index, tab ->
                    NavigationBarItem(
                        selected = selected == index,
                        onClick = { selected = index },
                        icon = { Icon(tab.icon, contentDescription = tab.label) },
                        label = { Text(tab.label, fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Theme.TileInk,
                            selectedTextColor = Theme.TileInk,
                            indicatorColor = Theme.Tile,
                            unselectedIconColor = Theme.InkSoft,
                            unselectedTextColor = Theme.InkSoft,
                        ),
                    )
                }
            }
        },
    ) { innerPadding ->
        // Box applies the scaffold insets so screen content clears the navigation bar.
        androidx.compose.foundation.layout.Box(
            modifier = Modifier.fillMaxSize().padding(innerPadding),
        ) {
            when (selected) {
                0 -> EmotionScreen(vm.emotion)
                1 -> RespiratoryScreen(vm.yamnet)
                else -> AboutScreen()
            }
        }
    }
}
