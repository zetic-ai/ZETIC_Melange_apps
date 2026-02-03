import Foundation

public struct CocoConfig {
    static func readClasses() -> [String] {
        guard let filepath = Bundle.main.path(forResource: "coco", ofType: "yaml") else {
            print("Could not find coco.yaml in bundle, using default classes")
            return []
        }
        
        do {
            let content = try String(contentsOfFile: filepath, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines)
            
            // Find where the names section starts
            guard let namesIndex = lines.firstIndex(where: { $0.trimmingCharacters(in: .whitespaces) == "names:" }) else {
                print("Could not find names section in YAML, using default classes")
                return []
            }
            
            var classes: [String] = Array(repeating: "", count: 80) // COCO has 80 classes
            
            // Parse the numbered class entries
            for i in (namesIndex + 1)..<lines.count {
                let line = lines[i].trimmingCharacters(in: .whitespaces)
                
                // Stop when we reach an empty line or a line that doesn't start with a number
                if line.isEmpty || !line.contains(":") {
                    break
                }
                
                // Parse lines like "  0: person"
                let parts = line.components(separatedBy: ":")
                if parts.count == 2,
                   let index = Int(parts[0].trimmingCharacters(in: .whitespaces)),
                   index >= 0 && index < 80 {
                    let className = parts[1].trimmingCharacters(in: .whitespaces)
                    classes[index] = className
                }
            }
            
            // Verify all classes were found
            guard !classes.contains("") else {
                print("Some classes were missing in the YAML, using default classes")
                return []
            }
            
            return classes
            
        } catch {
            print("Error reading YAML file: \(error.localizedDescription), using default classes")
            return []
        }
    }
}
