// Unlicense — cochranblock.org
// Pixel Forge iOS — thin Swift bridge to Rust binary.

import UIKit

// Rust FFI — calls into libpixel_forge_ios.a
@_silgen_name("pixel_forge_main")
func pixelForgeMain()

@_silgen_name("pixel_forge_version")
func pixelForgeVersion() -> UnsafePointer<CChar>

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // Initialize Rust side — extracts models, sets up paths
        pixelForgeMain()

        let version = String(cString: pixelForgeVersion())
        print("Pixel Forge iOS v\(version) started")

        // egui rendering happens via Metal on iOS
        // The Rust eframe handles window creation and rendering
        return true
    }
}
