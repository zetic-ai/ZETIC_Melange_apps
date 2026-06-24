package ai.zetic.demo.offlinetranslator.service

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Live network reachability — the Android equivalent of iOS `NWPathMonitor`. Drives the
 * "Online"/"Offline" status badge. Online is incidental; the demo's point is that translation
 * still works when this reads "Offline".
 */
class NetworkMonitor(context: Context) {
    private val connectivity =
        context.applicationContext.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

    private val _isOnline = MutableStateFlow(currentlyOnline())
    val isOnline: StateFlow<Boolean> = _isOnline.asStateFlow()

    private val callback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            _isOnline.value = true
        }

        override fun onLost(network: Network) {
            // Recompute against any remaining networks rather than assuming offline.
            _isOnline.value = currentlyOnline()
        }

        override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) {
            _isOnline.value = caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
        }
    }

    fun start() {
        runCatching { connectivity.registerDefaultNetworkCallback(callback) }
        _isOnline.value = currentlyOnline()
    }

    fun stop() {
        runCatching { connectivity.unregisterNetworkCallback(callback) }
    }

    private fun currentlyOnline(): Boolean {
        val network = connectivity.activeNetwork ?: return false
        val caps = connectivity.getNetworkCapabilities(network) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }
}
