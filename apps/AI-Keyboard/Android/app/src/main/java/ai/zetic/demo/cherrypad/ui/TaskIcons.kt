package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.model.KeyboardTask
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Autorenew
import androidx.compose.material.icons.filled.Forum
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.Verified
import androidx.compose.ui.graphics.vector.ImageVector

/** Maps each task to a Material icon (the Android analog of the iOS SF Symbols). */
val KeyboardTask.icon: ImageVector
    get() = when (this) {
        KeyboardTask.REWRITE -> Icons.Filled.Autorenew
        KeyboardTask.REPLY -> Icons.Filled.Forum
        KeyboardTask.TRANSLATE -> Icons.Filled.Language
        KeyboardTask.GRAMMAR -> Icons.Filled.Verified
    }
