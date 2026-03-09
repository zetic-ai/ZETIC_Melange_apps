package com.yeonseok.melangecounsel.data.repository

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.yeonseok.melangecounsel.domain.DiagnosticsSnapshot
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.diagnosticsDataStore: DataStore<Preferences> by preferencesDataStore(name = "diagnostics")

class DiagnosticsRepository(
    private val context: Context
) {
    private object Keys {
        val lastRunTimestamp = longPreferencesKey("last_run_timestamp")
        val lastGenerationMs = longPreferencesKey("last_generation_ms")
        val lastTokenCount = intPreferencesKey("last_token_count")
        val lastRawLog = stringPreferencesKey("last_raw_log")
        val lastStopReason = stringPreferencesKey("last_stop_reason")
    }

    val snapshotFlow: Flow<DiagnosticsSnapshot> = context.diagnosticsDataStore.data.map { preferences ->
        DiagnosticsSnapshot(
            lastRunTimestamp = preferences[Keys.lastRunTimestamp] ?: 0L,
            lastGenerationMs = preferences[Keys.lastGenerationMs] ?: 0L,
            lastTokenCount = preferences[Keys.lastTokenCount] ?: 0,
            lastRawLog = preferences[Keys.lastRawLog] ?: "No generation yet.",
            lastStopReason = preferences[Keys.lastStopReason] ?: "idle"
        )
    }

    suspend fun update(snapshot: DiagnosticsSnapshot) {
        context.diagnosticsDataStore.edit { preferences ->
            preferences[Keys.lastRunTimestamp] = snapshot.lastRunTimestamp
            preferences[Keys.lastGenerationMs] = snapshot.lastGenerationMs
            preferences[Keys.lastTokenCount] = snapshot.lastTokenCount
            preferences[Keys.lastRawLog] = snapshot.lastRawLog.take(5000)
            preferences[Keys.lastStopReason] = snapshot.lastStopReason
        }
    }
}
