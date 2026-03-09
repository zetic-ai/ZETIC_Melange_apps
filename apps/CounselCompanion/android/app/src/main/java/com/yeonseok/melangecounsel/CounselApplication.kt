package com.yeonseok.melangecounsel

import android.app.Application

class CounselApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        AppContainer.init(this)
    }
}
