import Flutter
import UIKit
import Vision

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "PlateOcrPlugin") {
      PlateOcrPlugin.register(with: registrar)
    }
  }
}

/// On-device license-plate OCR via Apple Vision (`VNRecognizeTextRequest`).
///
/// Dart sends a tightly-packed RGBA crop of a single detected plate (raw bytes +
/// width/height) over the `platehawk/ocr` MethodChannel. Vision runs on a
/// background queue (never blocking the platform/UI thread) and the recognized
/// strings + confidences are returned via the channel result on the main thread.
/// The Melange detection model is untouched — this is a separate on-device step.
public class PlateOcrPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(
      name: "platehawk/ocr",
      binaryMessenger: registrar.messenger()
    )
    let instance = PlateOcrPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  private let queue = DispatchQueue(label: "platehawk.ocr", qos: .userInitiated)

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard call.method == "recognize" else {
      result(FlutterMethodNotImplemented)
      return
    }
    guard
      let args = call.arguments as? [String: Any],
      let data = args["rgba"] as? FlutterStandardTypedData,
      let width = args["width"] as? Int,
      let height = args["height"] as? Int,
      width > 0, height > 0
    else {
      result(FlutterError(code: "bad_args",
                          message: "expected rgba (Uint8List), width, height",
                          details: nil))
      return
    }
    let bytes = data.data
    queue.async {
      let recognized = PlateOcrPlugin.recognize(rgba: bytes, width: width, height: height)
      DispatchQueue.main.async { result(recognized) }
    }
  }

  /// Build a CGImage from tightly-packed RGBA and run Vision text recognition.
  /// Returns `[{ "text": String, "confidence": Double }, ...]`.
  static func recognize(rgba: Data, width: Int, height: Int) -> [[String: Any]] {
    let bytesPerRow = width * 4
    guard rgba.count >= bytesPerRow * height else { return [] }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    // RGBA bytes: the 4th byte (alpha) is ignored (RGBX / noneSkipLast).
    let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue
    var pixels = [UInt8](rgba)
    guard
      let ctx = CGContext(
        data: &pixels,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo
      ),
      let cgImage = ctx.makeImage()
    else { return [] }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
      try handler.perform([request])
    } catch {
      return []
    }

    guard let observations = request.results else { return [] }
    var out: [[String: Any]] = []
    for obs in observations {
      guard let candidate = obs.topCandidates(1).first else { continue }
      out.append([
        "text": candidate.string,
        "confidence": Double(candidate.confidence),
      ])
    }
    return out
  }
}
