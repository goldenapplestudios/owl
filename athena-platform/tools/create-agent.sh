#!/bin/bash

set -e

# Function to display usage
usage() {
    echo "Usage: $0 --name <agent_name> --type <agent_type>"
    echo "Agent types: malware-re, threat-analyst, secure-dev, security-tester, architect, sre-security"
    exit 1
}

# Parse command line arguments
AGENT_NAME=""
AGENT_TYPE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            AGENT_NAME="$2"
            shift 2
            ;;
        --type)
            AGENT_TYPE="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Validate inputs
if [ -z "$AGENT_NAME" ] || [ -z "$AGENT_TYPE" ]; then
    usage
fi

# Map agent type to enum value and description
case $AGENT_TYPE in
    "malware-re")
        ENUM_TYPE="Doru"
        DESCRIPTION="Malware Reverse Engineering Agent"
        ;;
    "threat-analyst")
        ENUM_TYPE="Aegis"
        DESCRIPTION="Threat Analysis Agent"
        ;;
    "secure-dev")
        ENUM_TYPE="Forge"
        DESCRIPTION="Secure Development Agent"
        ;;
    "security-tester")
        ENUM_TYPE="Owl"
        DESCRIPTION="Security Testing Agent"
        ;;
    "architect")
        ENUM_TYPE="Weaver"
        DESCRIPTION="Security Architecture Agent"
        ;;
    "sre-security")
        ENUM_TYPE="Polis"
        DESCRIPTION="SRE Security Agent"
        ;;
    *)
        echo "Invalid agent type: $AGENT_TYPE"
        usage
        ;;
esac

# Convert agent name to proper case
AGENT_NAME_PROPER=$(echo "$AGENT_NAME" | sed 's/\b\(.\)/\u\1/g')

# Create target directory
TARGET_DIR="../athena-$AGENT_NAME"
if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR already exists!"
    exit 1
fi

echo "Creating agent: $AGENT_NAME_PROPER ($ENUM_TYPE)"

# Copy template
cp -r templates/agent-template "$TARGET_DIR"

# Replace placeholders in all files
find "$TARGET_DIR" -type f -name "*.toml" -o -name "*.rs" -o -name "*.md" | while read -r file; do
    sed -i.bak "s/{{agent_name}}/$AGENT_NAME/g" "$file"
    sed -i.bak "s/{{AgentName}}/$AGENT_NAME_PROPER/g" "$file"
    sed -i.bak "s/{{AgentType}}/$ENUM_TYPE/g" "$file"
    sed -i.bak "s/{{AgentDescription}}/$DESCRIPTION/g" "$file"
    rm "${file}.bak"
done

# Create agent-specific modules
mkdir -p "$TARGET_DIR/src/models"
mkdir -p "$TARGET_DIR/src/processors"
mkdir -p "$TARGET_DIR/src/analyzers"

# Initialize git repository
cd "$TARGET_DIR"
git init
git add .
git commit -m "Initial commit for $AGENT_NAME_PROPER agent"

echo "Agent created successfully at: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "1. cd $TARGET_DIR"
echo "2. cargo build --target wasm32-wasi --release"
echo "3. spin up"