#!/bin/bash

# Information about where to get the key
echo "You can get your ZETIC.MLange Key from https://mlange.zetic.ai"
echo "Please enter your ZETIC.MLange Key:"
read NEW_KEY

if [ -z "$NEW_KEY" ]; then
    echo "Error: Key cannot be empty."
    exit 1
fi

echo "Updating keys to: $NEW_KEY"

# Update iOS files (Swift) - looking for tokenKey: "..." 
# Also update privateTokenKey pattern and YOUR_MLANGE_KEY pattern
find apps -name "*.swift" -print0 | xargs -0 perl -i -pe 's/(tokenKey|privateTokenKey):\s*"[^"]*"/tokenKey: "'"$NEW_KEY"'"/g; s/"YOUR_MLANGE_KEY"/"'$NEW_KEY'"/g; s/"YOUR_PERSONAL_ACCESS_TOKEN"/"'$NEW_KEY'"/g'

# Update iOS Xcode scheme files (xcscheme) - EnvironmentVariable for ZETIC_ACCESS_TOKEN
find apps -name "*.xcscheme" -print0 | xargs -0 perl -i -pe 's/(key = "ZETIC_ACCESS_TOKEN"[^>]*value = ")[^"]*"/${1}"'"$NEW_KEY"'"/g'

# Update Android files (Kotlin/Java)
# Pattern 1: ZeticMLangeModel(context, "KEY", ...)
# Pattern 2: const val MLANGE_PERSONAL_ACCESS_TOKEN = "..."
# Pattern 3: val tokenKey = "..." (for t5_base_grammar_correction)
find apps -name "*.kt" -print0 | xargs -0 perl -0777 -i -pe 's/(ZeticMLangeModel\(\s*[^,]+,\s*)"[^"]*"/${1}"'"$NEW_KEY"'"/g'
find apps -name "*.java" -print0 | xargs -0 perl -0777 -i -pe 's/(ZeticMLangeModel\(\s*[^,]+,\s*)"[^"]*"/${1}"'"$NEW_KEY"'"/g'
# Update Constants.kt pattern
find apps -name "*.kt" -print0 | xargs -0 perl -i -pe 's/(MLANGE_PERSONAL_ACCESS_TOKEN\s*=\s*)"[^"]*"/${1}"'"$NEW_KEY"'"/g'
# Update tokenKey variable pattern
find apps -name "*.kt" -print0 | xargs -0 perl -i -pe 's/(val\s+tokenKey\s*=\s*)"[^"]*"/${1}"'"$NEW_KEY"'"/g'
# Update personalKey pattern (Qwen3Chat + Brew-AI-Notes Android). Matches the
# assignment/named-arg only — never the PLACEHOLDER_KEY constant used to detect
# an unconfigured key.
find apps -name "*.kt" -print0 | xargs -0 perl -i -pe 's/(personalKey\s*=\s*)"[^"]*"/${1}"'"$NEW_KEY"'"/g'
# Update any remaining YOUR_MLANGE_KEY placeholder in Kotlin/Java (e.g. multimodal-screener's
# PERSONAL_KEY constant), mirroring the Swift placeholder substitution above.
find apps -name "*.kt" -print0 | xargs -0 perl -i -pe 's/"YOUR_MLANGE_KEY"/"'"$NEW_KEY"'"/g'
find apps -name "*.java" -print0 | xargs -0 perl -i -pe 's/"YOUR_MLANGE_KEY"/"'"$NEW_KEY"'"/g'

# Update Flutter/Dart source-key placeholders without touching sentinel checks.
find apps -name "*.dart" -print0 | xargs -0 perl -i -pe 's/(const\s+_sourceMlangeKey\s*=\s*)"YOUR_MLANGE_KEY"/${1}"'"$NEW_KEY"'"/g'

echo "Key update complete!"
