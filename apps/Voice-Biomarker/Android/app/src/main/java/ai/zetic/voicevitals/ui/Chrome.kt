package ai.zetic.voicevitals.ui

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AirplanemodeActive
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

/**
 * "Nothing leaves this phone" — the privacy pitch, styled as a soft card with a
 * live network indicator so you can prove it still works in Airplane Mode.
 */
@Composable
fun PrivacyBanner() {
    val online by rememberIsOnline()
    Row(
        modifier = Modifier.card(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        IconTile(icon = Icons.Filled.Shield, size = 44)
        Spacer(Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text("100% on-device", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Theme.Ink)
            Spacer(Modifier.height(2.dp))
            Text("Nothing leaves this phone", fontSize = 12.sp, color = Theme.InkSoft)
        }
        Icon(
            imageVector = if (online) Icons.Filled.Wifi else Icons.Filled.AirplanemodeActive,
            contentDescription = null,
            tint = if (online) Theme.InkSoft else Theme.TileInk,
            modifier = Modifier.size(16.dp),
        )
        Spacer(Modifier.width(4.dp))
        Text(
            if (online) "Online" else "Offline",
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold,
            color = if (online) Theme.InkSoft else Theme.TileInk,
        )
    }
}

/** On-device inference latency chip. */
@Composable
fun LatencyBadge(ms: Double?) {
    Row(
        modifier = Modifier
            .clip(CircleShape)
            .background(Theme.Bg)
            .padding(horizontal = 11.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Filled.Bolt, contentDescription = null, tint = Theme.Warn, modifier = Modifier.size(13.dp))
        Spacer(Modifier.width(6.dp))
        if (ms != null) {
            Text("${ms.roundToInt()} ms", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Theme.Ink)
            Spacer(Modifier.width(4.dp))
            Text("on-device", fontSize = 10.sp, color = Theme.InkSoft)
        } else {
            Text("—", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Theme.InkSoft)
        }
    }
}

/** Lightweight reachability used only to display online/offline state. */
@Composable
fun rememberIsOnline(): State<Boolean> {
    val context = LocalContext.current
    val state = remember { mutableStateOf(true) }
    DisposableEffect(context) {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        fun current(): Boolean {
            val caps = cm.getNetworkCapabilities(cm.activeNetwork)
            return caps?.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
        }
        state.value = current()
        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                state.value = true
            }

            override fun onLost(network: Network) {
                state.value = current()
            }
        }
        cm.registerDefaultNetworkCallback(callback)
        onDispose { cm.unregisterNetworkCallback(callback) }
    }
    return state
}
