package com.yeonseok.melangecounsel.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ChatBubbleOutline
import androidx.compose.material.icons.rounded.History
import androidx.compose.material.icons.rounded.MonitorHeart
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.yeonseok.melangecounsel.ui.screens.ChatScreen
import com.yeonseok.melangecounsel.ui.screens.DiagnosticsScreen
import com.yeonseok.melangecounsel.ui.screens.SessionsScreen
import com.yeonseok.melangecounsel.ui.screens.SettingsScreen
import com.yeonseok.melangecounsel.ui.viewmodel.CounselUiState
import com.yeonseok.melangecounsel.ui.viewmodel.CounselViewModel

private enum class AppDestination(
    val route: String,
    val label: String,
    val icon: ImageVector
) {
    CHAT("chat", "Chat", Icons.Rounded.ChatBubbleOutline),
    SESSIONS("sessions", "Sessions", Icons.Rounded.History),
    SETTINGS("settings", "Settings", Icons.Rounded.Settings),
    DIAGNOSTICS("diagnostics", "Diagnostics", Icons.Rounded.MonitorHeart)
}

@Composable
fun AppRoot(state: CounselUiState, viewModel: CounselViewModel) {
    val navController = rememberNavController()
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(state.errorMessage) {
        state.errorMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.consumeError()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        bottomBar = { AppBottomBar(navController = navController) },
        containerColor = MaterialTheme.colorScheme.surface
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            MaterialTheme.colorScheme.surfaceContainerLowest,
                            MaterialTheme.colorScheme.surface
                        )
                    )
                )
                .padding(innerPadding)
        ) {
            AppNavHost(navController = navController, state = state, viewModel = viewModel)

            AnimatedVisibility(
                visible = state.isGenerating,
                enter = fadeIn() + slideInVertically(),
                exit = fadeOut() + slideOutVertically(),
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 12.dp)
            ) {
                Row(
                    modifier = Modifier
                        .shadow(8.dp, RoundedCornerShape(24.dp))
                        .background(
                            MaterialTheme.colorScheme.primaryContainer,
                            RoundedCornerShape(24.dp)
                        )
                        .padding(horizontal = 20.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = state.loadingMessage,
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }
            }
        }
    }
}

@Composable
private fun AppBottomBar(navController: NavHostController) {
    val destinations = AppDestination.entries
    val currentRoute = navController.currentBackStackEntryAsState().value?.destination?.route

    NavigationBar(
        containerColor = MaterialTheme.colorScheme.surfaceContainerLowest,
        tonalElevation = 0.dp
    ) {
        destinations.forEach { destination ->
            val selected = currentRoute == destination.route
            NavigationBarItem(
                selected = selected,
                onClick = {
                    navController.navigate(destination.route) {
                        popUpTo(destinations.first().route) { saveState = true }
                        launchSingleTop = true
                        restoreState = true
                    }
                },
                icon = {
                    Icon(
                        imageVector = destination.icon,
                        contentDescription = destination.label
                    )
                },
                label = {
                    Text(
                        destination.label,
                        style = MaterialTheme.typography.labelMedium
                    )
                },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = MaterialTheme.colorScheme.primary,
                    selectedTextColor = MaterialTheme.colorScheme.primary,
                    indicatorColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f),
                    unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant
                )
            )
        }
    }
}

@Composable
private fun AppNavHost(
    navController: NavHostController,
    state: CounselUiState,
    viewModel: CounselViewModel
) {
    NavHost(
        navController = navController,
        startDestination = AppDestination.CHAT.route,
        modifier = Modifier.fillMaxSize()
    ) {
        composable(AppDestination.CHAT.route) {
            ChatScreen(state = state, viewModel = viewModel)
        }
        composable(AppDestination.SESSIONS.route) {
            SessionsScreen(state = state, onSelectSession = viewModel::selectSession)
        }
        composable(AppDestination.SETTINGS.route) {
            SettingsScreen(state = state, viewModel = viewModel)
        }
        composable(AppDestination.DIAGNOSTICS.route) {
            DiagnosticsScreen(state = state)
        }
    }
}
