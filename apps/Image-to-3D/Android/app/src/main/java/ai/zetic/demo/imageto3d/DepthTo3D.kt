package ai.zetic.demo.imageto3d

import android.graphics.Bitmap
import kotlin.math.tan

/** Geometry buffers for one reconstruction — mirrors iOS DepthTo3D exactly. */
class MeshData(
    val positions: FloatArray,       // xyz per vertex
    val uvs: FloatArray,             // uv per vertex
    val triangleIndices: IntArray,
    val pointIndices: IntArray,      // point-cloud subset (edge-band dropped, 2× thinned)
    val pointColors: FloatArray,     // rgb per vertex
    val center: FloatArray,          // xyz
    val radius: Float,
)

/**
 * Converts a relative inverse-depth map into a planar-relief mesh:
 * x/y fixed on a reference plane (photo keeps its shape), z displaced by
 * smoothed disparity; quads across RAW-disparity discontinuities are culled,
 * the cut eroded by one quad, and the mask despeckled.
 */
object DepthTo3D {
    fun build(depth: DepthMap, texture: Bitmap): MeshData {
        val size = depth.width
        val stride = AppConfig.MESH_STRIDE
        val grid = (size - 1) / stride + 1

        val focal = size / 2f / tan(Math.toRadians(AppConfig.FOV_Y_DEGREES / 2.0)).toFloat()
        val center = (size - 1) / 2f
        val unitsPerPixel = AppConfig.PLANE_Z / focal
        val relief = AppConfig.RELIEF_DEPTH * size * unitsPerPixel

        // Raw disparity on the vertex grid; smoothed copy for geometry only.
        val raw = FloatArray(grid * grid)
        for (gy in 0 until grid) {
            val v = gy * stride
            for (gx in 0 until grid) {
                raw[gy * grid + gx] = depth.normalized(v * size + gx * stride)
            }
        }
        val disparity = gaussianSmoothed(raw, grid)

        val positions = FloatArray(grid * grid * 3)
        val uvs = FloatArray(grid * grid * 2)
        for (gy in 0 until grid) {
            val v = gy * stride
            for (gx in 0 until grid) {
                val u = gx * stride
                val i = gy * grid + gx
                val d = disparity[i]
                positions[i * 3] = (u - center) * unitsPerPixel
                positions[i * 3 + 1] = -(v - center) * unitsPerPixel
                positions[i * 3 + 2] = -(AppConfig.PLANE_Z + (0.5f - d) * relief)
                uvs[i * 2] = u / (size - 1f)
                uvs[i * 2 + 1] = v / (size - 1f)
            }
        }

        // Quad validity from RAW disparity, then erode + despeckle.
        val qside = grid - 1
        var valid = BooleanArray(qside * qside)
        val threshold = AppConfig.EDGE_THRESHOLD
        for (gy in 0 until qside) {
            for (gx in 0 until qside) {
                val i00 = gy * grid + gx
                val d0 = raw[i00]
                val d1 = raw[i00 + 1]
                val d2 = raw[i00 + grid]
                val d3 = raw[i00 + grid + 1]
                val hi = maxOf(d0, d1, d2, d3)
                val lo = minOf(d0, d1, d2, d3)
                valid[gy * qside + gx] = (hi - lo) <= threshold
            }
        }
        valid = erodedValid(valid, qside)
        valid = cleaned(valid, qside)

        val indices = ArrayList<Int>(qside * qside * 6)
        for (gy in 0 until qside) {
            for (gx in 0 until qside) {
                if (!valid[gy * qside + gx]) continue
                val i00 = gy * grid + gx
                val i10 = i00 + 1
                val i01 = i00 + grid
                val i11 = i01 + 1
                indices.add(i00); indices.add(i01); indices.add(i10)
                indices.add(i10); indices.add(i01); indices.add(i11)
            }
        }

        // Point cloud: 2× thinned; drop vertices whose every adjacent quad is culled.
        val pointIndices = ArrayList<Int>(grid * grid / 4)
        for (gy in 0 until grid step 2) {
            for (gx in 0 until grid step 2) {
                var touched = false
                for ((qy, qx) in arrayOf(gy - 1 to gx - 1, gy - 1 to gx, gy to gx - 1, gy to gx)) {
                    if (qy in 0 until qside && qx in 0 until qside && valid[qy * qside + qx]) {
                        touched = true
                        break
                    }
                }
                if (touched) pointIndices.add(gy * grid + gx)
            }
        }

        // Per-vertex colors sampled from the texture.
        val texPixels = IntArray(size * size)
        texture.getPixels(texPixels, 0, size, 0, 0, size, size)
        val colors = FloatArray(grid * grid * 3)
        for (gy in 0 until grid) {
            val v = gy * stride
            for (gx in 0 until grid) {
                val p = texPixels[v * size + gx * stride]
                val i = gy * grid + gx
                colors[i * 3] = ((p shr 16) and 0xFF) / 255f
                colors[i * 3 + 1] = ((p shr 8) and 0xFF) / 255f
                colors[i * 3 + 2] = (p and 0xFF) / 255f
            }
        }

        // Bounds over triangle-referenced vertices.
        var loX = Float.MAX_VALUE; var loY = Float.MAX_VALUE; var loZ = Float.MAX_VALUE
        var hiX = -Float.MAX_VALUE; var hiY = -Float.MAX_VALUE; var hiZ = -Float.MAX_VALUE
        val refs = if (indices.isEmpty()) (0 until grid * grid) else indices
        for (i in refs) {
            val x = positions[i * 3]; val y = positions[i * 3 + 1]; val z = positions[i * 3 + 2]
            if (x < loX) loX = x; if (x > hiX) hiX = x
            if (y < loY) loY = y; if (y > hiY) hiY = y
            if (z < loZ) loZ = z; if (z > hiZ) hiZ = z
        }
        val dx = hiX - loX; val dy = hiY - loY; val dz = hiZ - loZ
        val radius = (Math.sqrt((dx * dx + dy * dy + dz * dz).toDouble()) / 2).toFloat()

        return MeshData(
            positions = positions,
            uvs = uvs,
            triangleIndices = indices.toIntArray(),
            pointIndices = pointIndices.toIntArray(),
            pointColors = colors,
            center = floatArrayOf((loX + hiX) / 2, (loY + hiY) / 2, (loZ + hiZ) / 2),
            radius = maxOf(radius, 0.05f),
        )
    }

    /** Separable 5-tap Gaussian ([1,4,6,4,1]/16) over the vertex grid. */
    private fun gaussianSmoothed(input: FloatArray, grid: Int): FloatArray {
        val k = floatArrayOf(1 / 16f, 4 / 16f, 6 / 16f, 4 / 16f, 1 / 16f)
        val tmp = FloatArray(input.size)
        val out = FloatArray(input.size)
        for (gy in 0 until grid) {
            for (gx in 0 until grid) {
                var acc = 0f
                for (t in -2..2) {
                    val x = (gx + t).coerceIn(0, grid - 1)
                    acc += input[gy * grid + x] * k[t + 2]
                }
                tmp[gy * grid + gx] = acc
            }
        }
        for (gy in 0 until grid) {
            for (gx in 0 until grid) {
                var acc = 0f
                for (t in -2..2) {
                    val y = (gy + t).coerceIn(0, grid - 1)
                    acc += tmp[y * grid + gx] * k[t + 2]
                }
                out[gy * grid + gx] = acc
            }
        }
        return out
    }

    /** Valid only when the full 8-neighborhood is valid (dilates the cut). */
    private fun erodedValid(mask: BooleanArray, side: Int): BooleanArray {
        val out = mask.copyOf()
        for (y in 0 until side) {
            for (x in 0 until side) {
                if (!mask[y * side + x]) continue
                var keep = true
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        val ny = (y + dy).coerceIn(0, side - 1)
                        val nx = (x + dx).coerceIn(0, side - 1)
                        if (!mask[ny * side + nx]) keep = false
                    }
                }
                out[y * side + x] = keep
            }
        }
        return out
    }

    /** Majority filter: flip quads whose 3×3 neighborhood vote is lopsided. */
    private fun cleaned(mask: BooleanArray, side: Int): BooleanArray {
        val out = mask.copyOf()
        for (y in 0 until side) {
            for (x in 0 until side) {
                var votes = 0
                var total = 0
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val ny = y + dy
                        val nx = x + dx
                        if (ny !in 0 until side || nx !in 0 until side) continue
                        total++
                        if (mask[ny * side + nx]) votes++
                    }
                }
                val idx = y * side + x
                if (mask[idx] && votes <= total / 4) out[idx] = false
                if (!mask[idx] && votes >= total - total / 4) out[idx] = true
            }
        }
        return out
    }
}
