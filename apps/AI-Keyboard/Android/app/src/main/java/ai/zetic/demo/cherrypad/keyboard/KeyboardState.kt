package ai.zetic.demo.cherrypad.keyboard

import ai.zetic.demo.cherrypad.model.KeyboardTask
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

enum class KeyPlane { LETTERS, NUMBERS, SYMBOLS }

/** Observable keyboard UI state (mirrors iOS `KeyboardState`). */
class KeyboardState {
    var plane by mutableStateOf(KeyPlane.LETTERS)
    var shifted by mutableStateOf(true)          // starts uppercase, like iOS
    var processing by mutableStateOf(false)
    var statusText by mutableStateOf<String?>(null)
    var resultText by mutableStateOf<String?>(null)
    var activeTask by mutableStateOf<KeyboardTask?>(null)
    var banner by mutableStateOf<String?>(null)
}

/** Text-editing + AI actions the keyboard view delegates to the IME service. */
interface KeyboardActions {
    fun insert(text: String)
    fun deleteBackward()
    fun newLine()
    fun nextKeyboard()
    fun runAction(task: KeyboardTask)
    fun insertResult()
    fun dismissResult()
}
