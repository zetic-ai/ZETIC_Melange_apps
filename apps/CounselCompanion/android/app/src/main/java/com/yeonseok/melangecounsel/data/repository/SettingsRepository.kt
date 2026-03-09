package com.yeonseok.melangecounsel.data.repository

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.yeonseok.melangecounsel.ui.theme.ThemeMode
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

class SettingsRepository(
    private val context: Context
) {
    private object Keys {
        val themeMode = stringPreferencesKey("theme_mode")
        val systemPrompt = stringPreferencesKey("system_prompt")
    }

    val themeModeFlow: Flow<ThemeMode> = context.settingsDataStore.data.map { preferences ->
        runCatching {
            ThemeMode.valueOf(preferences[Keys.themeMode] ?: ThemeMode.SYSTEM.name)
        }.getOrDefault(ThemeMode.SYSTEM)
    }

    val systemPromptFlow: Flow<String> = context.settingsDataStore.data.map { preferences ->
        preferences[Keys.systemPrompt]
            ?: "You are a kind mental wellness companion. Listen actively, ask reflective questions, and avoid medical diagnosis."
    }

    suspend fun setThemeMode(mode: ThemeMode) {
        context.settingsDataStore.edit { preferences ->
            preferences[Keys.themeMode] = mode.name
        }
    }

    suspend fun setSystemPrompt(prompt: String) {
        context.settingsDataStore.edit { preferences ->
            preferences[Keys.systemPrompt] = prompt
        }
    }
}
