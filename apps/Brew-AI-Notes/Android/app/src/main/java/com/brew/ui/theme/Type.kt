package com.brew.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.sp

/** Body text uses the system sans (SF Pro on iOS → default sans here). */
val BrewTypography = Typography(
    bodyLarge = TextStyle(fontFamily = FontFamily.Default, fontSize = 17.sp),
    bodyMedium = TextStyle(fontFamily = FontFamily.Default, fontSize = 15.sp),
)
