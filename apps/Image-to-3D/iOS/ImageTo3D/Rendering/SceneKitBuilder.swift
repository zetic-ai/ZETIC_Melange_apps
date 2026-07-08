import SceneKit
import UIKit

/// Builds the SceneKit scene for the depth-relief reconstruction. Shared by
/// the interactive Model3DView and the headless self-test renderer.
enum SceneKitBuilder {
    struct Built {
        let scene: SCNScene
        let contentNode: SCNNode   // spin/orbit target (pivoted at content center)
        let meshNode: SCNNode
        let pointNode: SCNNode
        let cameraNode: SCNNode
    }

    // MARK: - Depth-relief preview (photo-textured)

    static func build(mesh: MeshData, texture: UIImage) -> Built {
        let vertexSource = SCNGeometrySource(vertices: mesh.positions.map {
            SCNVector3($0.x, $0.y, $0.z)
        })
        let uvSource = SCNGeometrySource(textureCoordinates: mesh.uvs.map {
            CGPoint(x: CGFloat($0.x), y: CGFloat($0.y))
        })

        let meshGeometry = SCNGeometry(sources: [vertexSource, uvSource],
                                       elements: [triangleElement(mesh.triangleIndices)])
        let material = SCNMaterial()
        material.diffuse.contents = texture
        material.diffuse.mipFilter = .linear
        material.lightingModel = .constant
        material.isDoubleSided = true
        meshGeometry.materials = [material]

        let pointGeometry = SCNGeometry(
            sources: [vertexSource, colorSource(mesh.pointColors, count: mesh.positions.count)],
            elements: [pointElement(mesh.pointIndices)])
        pointGeometry.firstMaterial?.lightingModel = .constant

        return assemble(meshGeometry: meshGeometry,
                        pointGeometry: pointGeometry,
                        center: mesh.center,
                        radius: mesh.radius)
    }

    /// Slow turntable oscillation: sway ±35° around Y so the 3D relief is
    /// obvious without ever showing the hollow back of a reconstruction.
    static func startTurntable(_ node: SCNNode) {
        let sway: CGFloat = 35 * .pi / 180
        let right = SCNAction.rotateBy(x: 0, y: sway, z: 0, duration: 4)
        right.timingMode = .easeInEaseOut
        let left = SCNAction.rotateBy(x: 0, y: -sway, z: 0, duration: 4)
        left.timingMode = .easeInEaseOut
        node.runAction(.repeatForever(.sequence([right, left, left, right])),
                       forKey: "turntable")
    }

    // MARK: - Shared pieces

    private static func assemble(meshGeometry: SCNGeometry,
                                 pointGeometry: SCNGeometry,
                                 center: SIMD3<Float>,
                                 radius: Float) -> Built {
        let scene = SCNScene()
        scene.background.contents = gradientImage()

        let meshNode = SCNNode(geometry: meshGeometry)
        let pointNode = SCNNode(geometry: pointGeometry)
        let contentNode = SCNNode()
        contentNode.addChildNode(meshNode)
        contentNode.addChildNode(pointNode)
        contentNode.pivot = SCNMatrix4MakeTranslation(center.x, center.y, center.z)
        scene.rootNode.addChildNode(contentNode)

        let camera = SCNCamera()
        camera.fieldOfView = CGFloat(AppConfig.fovYDegrees)
        let distance = radius / tan(AppConfig.fovYDegrees * .pi / 180 / 2) * 1.1
        camera.zNear = 0.01
        camera.zFar = Double(distance + radius) * 4
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        let elevation: Float = 0.18   // radians, ~10°
        cameraNode.position = SCNVector3(0, distance * sin(elevation), distance * cos(elevation))
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)

        return Built(scene: scene,
                     contentNode: contentNode,
                     meshNode: meshNode,
                     pointNode: pointNode,
                     cameraNode: cameraNode)
    }

    private static func triangleElement(_ indices: [UInt32]) -> SCNGeometryElement {
        let data = indices.withUnsafeBufferPointer { Data(buffer: $0) }
        return SCNGeometryElement(data: data,
                                  primitiveType: .triangles,
                                  primitiveCount: indices.count / 3,
                                  bytesPerIndex: MemoryLayout<UInt32>.size)
    }

    private static func pointElement(_ indices: [UInt32]) -> SCNGeometryElement {
        let data = indices.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(data: data,
                                         primitiveType: .point,
                                         primitiveCount: indices.count,
                                         bytesPerIndex: MemoryLayout<UInt32>.size)
        element.pointSize = 0.006
        element.minimumPointScreenSpaceRadius = 1.5
        element.maximumPointScreenSpaceRadius = 10
        return element
    }

    private static func colorSource(_ rgb: [Float], count: Int) -> SCNGeometrySource {
        let data = rgb.withUnsafeBufferPointer { Data(buffer: $0) }
        return SCNGeometrySource(data: data,
                                 semantic: .color,
                                 vectorCount: count,
                                 usesFloatComponents: true,
                                 componentsPerVector: 3,
                                 bytesPerComponent: MemoryLayout<Float>.size,
                                 dataOffset: 0,
                                 dataStride: MemoryLayout<Float>.size * 3)
    }

    /// Soft vertical gradient so occlusion holes read as shadow, not voids.
    private static func gradientImage() -> UIImage {
        let size = CGSize(width: 2, height: 256)
        return UIGraphicsImageRenderer(size: size).image { ctx in
            let colors = [UIColor(white: 0.16, alpha: 1).cgColor,
                          UIColor(white: 0.05, alpha: 1).cgColor]
            let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                                      colors: colors as CFArray,
                                      locations: [0, 1])!
            ctx.cgContext.drawLinearGradient(gradient,
                                             start: .zero,
                                             end: CGPoint(x: 0, y: size.height),
                                             options: [])
        }
    }
}
