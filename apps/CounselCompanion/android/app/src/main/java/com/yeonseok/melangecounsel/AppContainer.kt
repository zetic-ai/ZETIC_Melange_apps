package com.yeonseok.melangecounsel

import android.content.Context
import com.yeonseok.melangecounsel.data.local.AppDatabase
import com.yeonseok.melangecounsel.data.repository.ChatRepository
import com.yeonseok.melangecounsel.data.repository.DiagnosticsRepository
import com.yeonseok.melangecounsel.data.repository.SettingsRepository
import com.yeonseok.melangecounsel.llm.ZeticChatEngine
import com.yeonseok.melangecounsel.util.ConnectivityObserver

object AppContainer {
    private var initialized = false

    lateinit var chatRepository: ChatRepository
        private set
    lateinit var settingsRepository: SettingsRepository
        private set
    lateinit var diagnosticsRepository: DiagnosticsRepository
        private set
    lateinit var connectivityObserver: ConnectivityObserver
        private set
    lateinit var chatEngine: ZeticChatEngine
        private set

    fun init(context: Context) {
        if (initialized) return
        val appContext = context.applicationContext
        val database = AppDatabase.create(appContext)
        chatRepository = ChatRepository(database.chatDao())
        settingsRepository = SettingsRepository(appContext)
        diagnosticsRepository = DiagnosticsRepository(appContext)
        connectivityObserver = ConnectivityObserver(appContext)
        chatEngine = ZeticChatEngine(appContext)
        initialized = true
    }
}
