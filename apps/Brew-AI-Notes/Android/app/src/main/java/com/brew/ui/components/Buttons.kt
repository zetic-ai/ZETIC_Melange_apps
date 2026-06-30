package com.brew.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.brew.ui.theme.BrewColors

/** A circular icon button on an elevated/card fill (iOS `CircleIconButton`). */
@Composable
fun CircleIconButton(
    icon: ImageVector,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    diameter: Int = 46,
    iconSize: Int = 17,
    fill: Color = BrewColors.cardElevated,
    tint: Color = BrewColors.ink,
) {
    Box(
        modifier = modifier
            .size(diameter.dp)
            .clip(CircleShape)
            .background(fill)
            .clickable { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = null, tint = tint, modifier = Modifier.size(iconSize.dp))
    }
}
