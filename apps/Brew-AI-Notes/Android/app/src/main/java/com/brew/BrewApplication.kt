package com.brew

import android.app.Application
import com.brew.asr.TranscriptionWorker
import com.brew.data.NotesRepository
import com.brew.engine.ModelCoordinator

/** App-scoped singletons (lightweight service locator). */
class BrewApplication : Application() {
    lateinit var repository: NotesRepository
        private set
    lateinit var coordinator: ModelCoordinator
        private set
    lateinit var transcriptionWorker: TranscriptionWorker
        private set

    override fun onCreate() {
        super.onCreate()
        instance = this
        repository = NotesRepository(this)
        coordinator = ModelCoordinator(this)
        transcriptionWorker = TranscriptionWorker(repository, coordinator)
    }

    companion object {
        lateinit var instance: BrewApplication
            private set
    }
}
