#!/bin/bash

# Script to restore all API keys back to YOUR_MLANGE_KEY placeholder
# This reverses the changes made by adapt_mlange_key.sh

PLACEHOLDER="YOUR_MLANGE_KEY"

echo "Restoring all API keys to placeholder: $PLACEHOLDER"
echo ""

# Update iOS files (Swift) - restore tokenKey: "..." to YOUR_MLANGE_KEY
# This will match any actual key and replace with placeholder
find apps -name "*.swift" -print0 | xargs -0 perl -i -pe 's/(tokenKey|privateTokenKey):\s*"[^"]*"/tokenKey: "'"$PLACEHOLDER"'"/g'

# Swift: @Published var tokenKey = "..." (ChronosTimeSeries)
find apps -name "*.swift" -print0 | xargs -0 perl -i -pe 's/(@Published\s+var\s+tokenKey\s*=\s*)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'
# Swift: tokenKey = "..." (assignment, e.g. in init)
find apps -name "*.swift" -print0 | xargs -0 perl -i -pe 's/(tokenKey\s*=\s*)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'
# Swift: accessToken = ... ?? "..." (TextAnonymizer)
find apps -name "*.swift" -print0 | xargs -0 perl -i -pe 's/(accessToken\s*=\s*(?:.*?\?\?\s*)?)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'
# Swift: personalKey = ... ?? "..." (TextAnonymizer)
find apps -name "*.swift" -print0 | xargs -0 perl -i -pe 's/(personalKey\s*=\s*(?:.*?\?\?\s*)?)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'

# Update iOS Xcode scheme files (xcscheme) - restore ZETIC_ACCESS_TOKEN
find apps -name "*.xcscheme" -print0 | xargs -0 perl -i -pe 's/(key = "ZETIC_ACCESS_TOKEN"[^>]*value = ")[^"]*"/${1}"'"$PLACEHOLDER"'"/g'

# Update Android files (Kotlin/Java)
# Pattern 1: ZeticMLangeModel(context, "KEY", ...) - restore first parameter (API key)
find apps -name "*.kt" -print0 | xargs -0 perl -0777 -i -pe 's/(ZeticMLangeModel\(\s*[^,]+,\s*)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'
find apps -name "*.java" -print0 | xargs -0 perl -0777 -i -pe 's/(ZeticMLangeModel\(\s*[^,]+,\s*)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'

# Pattern 2: const val PERSONAL_KEY = "..."
find apps -name "*.kt" -print0 | xargs -0 perl -i -pe 's/(PERSONAL_KEY\s*=\s*)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'

# Pattern 3: val tokenKey = "..." (for t5_base_grammar_correction)
find apps -name "*.kt" -print0 | xargs -0 perl -i -pe 's/(val\s+tokenKey\s*=\s*)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'

# Pattern 3b: val token = "..." (ChronosTimeSeries loadModel)
find apps -name "*.kt" -print0 | xargs -0 perl -i -pe 's/(val\s+token\s*=\s*)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'

# Pattern 4: val mlangeKey = "..." (for t5_base_grammar_correction iOS)
find apps -name "*.swift" -print0 | xargs -0 perl -i -pe 's/(let\s+mlangeKey\s*=\s*)"[^"]*"/${1}"'"$PLACEHOLDER"'"/g'

echo "✅ All API keys have been restored to placeholder: $PLACEHOLDER"
echo ""
echo "Note: Git filter will automatically protect these keys on commit."
