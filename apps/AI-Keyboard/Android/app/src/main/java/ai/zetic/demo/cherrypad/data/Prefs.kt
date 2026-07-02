package ai.zetic.demo.cherrypad.data

import android.content.Context
import ai.zetic.demo.cherrypad.model.Language

/**
 * Small shared preference store. The container app and the keyboard run in the same
 * process, so a single `SharedPreferences` replaces the iOS App Group: the app writes the
 * chosen translate target here and the keyboard reads it.
 */
object Prefs {
    private const val FILE = "cherrypad.prefs"
    private const val KEY_TARGET_LANG = "cherrypad.targetLang" // stores Language.englishName
    private const val KEY_ONBOARDED = "hasSeenOnboarding"

    private fun prefs(context: Context) =
        context.applicationContext.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    /** The translate target's englishName; defaults to Korean, matching iOS. */
    fun targetLanguageName(context: Context): String =
        prefs(context).getString(KEY_TARGET_LANG, null) ?: Language.default.englishName

    fun setTargetLanguageName(context: Context, englishName: String) {
        prefs(context).edit().putString(KEY_TARGET_LANG, englishName).apply()
    }

    fun hasSeenOnboarding(context: Context): Boolean =
        prefs(context).getBoolean(KEY_ONBOARDED, false)

    fun setHasSeenOnboarding(context: Context, value: Boolean) {
        prefs(context).edit().putBoolean(KEY_ONBOARDED, value).apply()
    }
}
