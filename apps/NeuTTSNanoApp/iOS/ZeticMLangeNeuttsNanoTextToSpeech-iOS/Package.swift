// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NeuTTSNanoApp",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .executable(name: "NeuTTSNanoApp", targets: ["NeuTTSNanoApp"])
    ],
    dependencies: [
        .package(url: "https://github.com/zetic-ai/ZeticMLangeiOS.git", .upToNextMajor(from: "1.0.0"))
    ],
    targets: [
        .executableTarget(
            name: "NeuTTSNanoApp",
            dependencies: [
                .product(name: "ZeticMLange", package: "ZeticMLangeiOS")
            ]
        )
    ]
)
