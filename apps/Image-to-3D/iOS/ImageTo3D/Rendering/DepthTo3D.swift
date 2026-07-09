import Foundation
import UIKit
import simd

/// Geometry buffers for one reconstruction: a textured triangle grid and the
/// matching colored point cloud (same vertex positions).
struct MeshData {
    var positions: [SIMD3<Float>]
    var uvs: [SIMD2<Float>]
    var triangleIndices: [UInt32]
    var pointIndices: [UInt32]   // point-cloud subset (edge-band vertices dropped)
    var pointColors: [Float]     // packed RGB triplets (3 floats per vertex)
    var boundsMin: SIMD3<Float>
    var boundsMax: SIMD3<Float>

    var center: SIMD3<Float> { (boundsMin + boundsMax) / 2 }
    var radius: Float { simd_length(boundsMax - boundsMin) / 2 }
}

/// Converts a relative inverse-depth map into a 3D grid mesh via pinhole
/// unprojection. The camera sits at the origin looking down -Z, +Y up.
enum DepthTo3D {
    static func build(depth: DepthMap, texture: UIImage) -> MeshData {
        let size = depth.width
        let stride = AppConfig.meshStride
        let grid = (size - 1) / stride + 1

        // Planar-relief coordinates: x/y are fixed on the reference plane
        // (world units per source pixel), z is displaced by disparity.
        let focal = Float(size) / 2 / tan(AppConfig.fovYDegrees * .pi / 180 / 2)
        let center = Float(size - 1) / 2
        let unitsPerPixel = AppConfig.planeZ / focal
        let relief = AppConfig.reliefDepth * Float(size) * unitsPerPixel

        // Sample normalized disparity onto the vertex grid. Geometry uses a
        // smoothed copy (pleasant relief); the discontinuity test uses the RAW
        // values — smoothing spreads the fg/bg jump over several pixels, which
        // lets stretched "skirt" quads slip under the threshold and show up as
        // a hairy dark fringe around silhouettes.
        var rawDisparity = [Float](repeating: 0, count: grid * grid)
        for gy in 0..<grid {
            let v = gy * stride
            for gx in 0..<grid {
                rawDisparity[gy * grid + gx] = depth.normalized(v * size + gx * stride)
            }
        }
        let disparity = gaussianSmoothed(rawDisparity, grid: grid)

        let textureRGB = sampleTexture(texture, gridSize: grid, imageStride: stride)

        var positions = [SIMD3<Float>]()
        var uvs = [SIMD2<Float>]()
        positions.reserveCapacity(grid * grid)
        uvs.reserveCapacity(grid * grid)

        for gy in 0..<grid {
            let v = gy * stride
            for gx in 0..<grid {
                let u = gx * stride
                let d = disparity[gy * grid + gx]
                // Image v grows downward, scene Y grows upward → flip Y.
                // d = 1 (closest) pushes toward the camera, d = 0 away.
                positions.append(SIMD3<Float>((Float(u) - center) * unitsPerPixel,
                                              -(Float(v) - center) * unitsPerPixel,
                                              -(AppConfig.planeZ + (0.5 - d) * relief)))
                uvs.append(SIMD2<Float>(Float(u) / Float(size - 1),
                                        Float(v) / Float(size - 1)))
            }
        }

        // Quad validity from RAW disparity: a quad spanning a depth
        // discontinuity is culled whole. The cut is then dilated one quad so
        // anti-aliased edge pixels (the dark outline smear) go with it, and
        // despeckled so isolated spikes/pinholes vanish.
        let qside = grid - 1
        var valid = [Bool](repeating: false, count: qside * qside)
        let threshold = AppConfig.edgeThreshold
        for gy in 0..<qside {
            for gx in 0..<qside {
                let i00 = gy * grid + gx
                let d0 = rawDisparity[i00]
                let d1 = rawDisparity[i00 + 1]
                let d2 = rawDisparity[i00 + grid]
                let d3 = rawDisparity[i00 + grid + 1]
                let hi = max(max(d0, d1), max(d2, d3))
                let lo = min(min(d0, d1), min(d2, d3))
                valid[gy * qside + gx] = (hi - lo) <= threshold
            }
        }
        valid = erodedValid(valid, side: qside)
        valid = cleaned(valid, side: qside)

        var indices = [UInt32]()
        indices.reserveCapacity(qside * qside * 6)
        for gy in 0..<qside {
            for gx in 0..<qside {
                guard valid[gy * qside + gx] else { continue }
                let i00 = UInt32(gy * grid + gx)
                let i10 = i00 + 1
                let i01 = i00 + UInt32(grid)
                let i11 = i01 + 1
                indices.append(contentsOf: [i00, i01, i10, i10, i01, i11])
            }
        }

        // Point cloud: skip vertices whose every adjacent quad is culled —
        // they sit inside the smeared discontinuity band and render as a
        // noisy halo around silhouettes. Thinned 2× (full grid is moiré soup).
        var pointIndices = [UInt32]()
        pointIndices.reserveCapacity(grid * grid / 4)
        for gy in Swift.stride(from: 0, to: grid, by: 2) {
            for gx in Swift.stride(from: 0, to: grid, by: 2) {
                var touched = false
                for (qy, qx) in [(gy - 1, gx - 1), (gy - 1, gx), (gy, gx - 1), (gy, gx)] {
                    if qy >= 0, qy < qside, qx >= 0, qx < qside, valid[qy * qside + qx] {
                        touched = true
                        break
                    }
                }
                if touched { pointIndices.append(UInt32(gy * grid + gx)) }
            }
        }

        // Bounds over vertices actually referenced by triangles, so stray
        // unreferenced grid corners don't inflate the camera framing.
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        if indices.isEmpty {
            for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        } else {
            for i in indices {
                let p = positions[Int(i)]
                lo = simd_min(lo, p)
                hi = simd_max(hi, p)
            }
        }

        return MeshData(positions: positions,
                        uvs: uvs,
                        triangleIndices: indices,
                        pointIndices: pointIndices,
                        pointColors: textureRGB,
                        boundsMin: lo,
                        boundsMax: hi)
    }

    /// Separable 5-tap Gaussian ([1,4,6,4,1]/16) over the vertex grid.
    private static func gaussianSmoothed(_ input: [Float], grid: Int) -> [Float] {
        let kernel: [Float] = [1 / 16.0, 4 / 16.0, 6 / 16.0, 4 / 16.0, 1 / 16.0]
        var tmp = [Float](repeating: 0, count: input.count)
        var out = [Float](repeating: 0, count: input.count)
        for gy in 0..<grid {
            for gx in 0..<grid {
                var acc: Float = 0
                for k in -2...2 {
                    let x = min(max(gx + k, 0), grid - 1)
                    acc += input[gy * grid + x] * kernel[k + 2]
                }
                tmp[gy * grid + gx] = acc
            }
        }
        for gy in 0..<grid {
            for gx in 0..<grid {
                var acc: Float = 0
                for k in -2...2 {
                    let y = min(max(gy + k, 0), grid - 1)
                    acc += tmp[y * grid + gx] * kernel[k + 2]
                }
                out[gy * grid + gx] = acc
            }
        }
        return out
    }

    /// Erodes the valid region by one quad (a quad stays valid only if its
    /// full 8-neighborhood is valid) — dilates the discontinuity cut so the
    /// anti-aliased silhouette pixels fall inside it.
    private static func erodedValid(_ mask: [Bool], side: Int) -> [Bool] {
        var out = mask
        for y in 0..<side {
            for x in 0..<side {
                guard mask[y * side + x] else { continue }
                var keep = true
                for dy in -1...1 {
                    for dx in -1...1 {
                        let ny = min(max(y + dy, 0), side - 1)
                        let nx = min(max(x + dx, 0), side - 1)
                        if !mask[ny * side + nx] { keep = false }
                    }
                }
                out[y * side + x] = keep
            }
        }
        return out
    }

    /// Majority filter on the quad mask: a quad flips to match its 3×3
    /// neighborhood when the vote is lopsided. Removes one-quad sawtooth
    /// spikes along silhouettes and pinholes inside surfaces while leaving
    /// genuine edges alone.
    private static func cleaned(_ mask: [Bool], side: Int) -> [Bool] {
        var out = mask
        for y in 0..<side {
            for x in 0..<side {
                var votes = 0
                var total = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let ny = y + dy
                        let nx = x + dx
                        guard ny >= 0, ny < side, nx >= 0, nx < side,
                              !(dx == 0 && dy == 0) else { continue }
                        total += 1
                        if mask[ny * side + nx] { votes += 1 }
                    }
                }
                let idx = y * side + x
                if mask[idx] && votes <= total / 4 { out[idx] = false }
                if !mask[idx] && votes >= total - total / 4 { out[idx] = true }
            }
        }
        return out
    }

    /// Samples the 518×518 texture at each grid vertex → packed RGB floats.
    private static func sampleTexture(_ texture: UIImage, gridSize: Int, imageStride: Int) -> [Float] {
        let size = AppConfig.inputSize
        var colors = [Float](repeating: 0.8, count: gridSize * gridSize * 3)
        guard let cgImage = texture.cgImage else { return colors }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        let drawn: Bool = pixels.withUnsafeMutableBufferPointer { ptr in
            guard let context = CGContext(data: ptr.baseAddress,
                                          width: size,
                                          height: size,
                                          bitsPerComponent: 8,
                                          bytesPerRow: size * 4,
                                          space: colorSpace,
                                          bitmapInfo: bitmapInfo.rawValue) else { return false }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
            return true
        }
        guard drawn else { return colors }

        for gy in 0..<gridSize {
            let v = gy * imageStride
            for gx in 0..<gridSize {
                let u = gx * imageStride
                let src = (v * size + u) * 4
                let dst = (gy * gridSize + gx) * 3
                colors[dst] = Float(pixels[src]) / 255.0
                colors[dst + 1] = Float(pixels[src + 1]) / 255.0
                colors[dst + 2] = Float(pixels[src + 2]) / 255.0
            }
        }
        return colors
    }
}
