#!/bin/bash
# auto-commit.sh - Interactive commit message generator

# Color definitions
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

# Path configurations
AUTO_COMMIT_ROOT="path/to/your/auto-commit/dir"
PYTHON_CMD="$AUTO_COMMIT_ROOT/venv/bin/python"
MAIN_SCRIPT="$AUTO_COMMIT_ROOT/main.py"
TARGET_REPO=$(git rev-parse --show-toplevel 2>/dev/null)

# Check if in a Git repository
if [ -z "$TARGET_REPO" ]; then
    echo -e "${RED}Error: Current directory is not a Git repository${NC}"
    exit 1
fi

# Check required files
if [ ! -f "$PYTHON_CMD" ] || [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${RED}Error: Auto-commit tool not found${NC}"
    exit 1
fi

# Check for staged changes in target repo
if [ -z "$(git -C "$TARGET_REPO" diff --cached --name-only)" ]; then
    echo -e "${RED}Error: No staged changes in target repository${NC}"
    exit 1
fi

# Main interaction flow
while true; do
    clear
    echo -e "${CYAN}=== Auto Commit Message Generation ===${NC}"
    echo -e "Repository: ${YELLOW}$TARGET_REPO${NC}"
    echo -e "${CYAN}Generating message...${NC}"
    
    # Generate message
    # commit_msg=$("$PYTHON_CMD" "$MAIN_SCRIPT" --repo "$TARGET_REPO")
    commit_msg=$(source "$AUTO_COMMIT_ROOT/venv/bin/activate" && \
                "$PYTHON_CMD" "$MAIN_SCRIPT" --repo "$TARGET_REPO" 2>/dev/null)

    if [ -z "$commit_msg" ]; then
        echo -e "${RED}Error: Failed to generate commit message${NC}"
        exit 1
    fi
    
    # Display generated message
    echo -e "\n${GREEN}Generated Commit Message:${NC}"
    echo -e "$commit_msg" | fold -s -w 80
    echo -e "\n${CYAN}Options:${NC}"
    echo -e "  ${GREEN}Tab key${NC} - Accept and use this message"
    echo -e "  ${YELLOW}Enter key${NC} - Reject and regenerate"
    echo -e "  ${RED}Any other key${NC} - Exit"
    
    # Read single character (silent)
    IFS= read -rsn1 key
    # Improved key detection
    if [[ "$key" == "" ]]; then  # Enter key (empty string)
        echo -e "\n${YELLOW}Regenerating message...${NC}"
        sleep 1
        continue
    elif [[ "$key" == $'\t' ]]; then  # Tab key
        echo -e "\n${GREEN}Commit message accepted${NC}"
        if ! git -C "$TARGET_REPO" commit -m "$commit_msg"; then
            echo -e "${RED}Error: Commit failed${NC}"
            exit 1
        fi
        echo -e "${GREEN}Commit successful!${NC}"
        exit 0
    else  # Other keys
        echo -e "\n${RED}Operation cancelled${NC}"
        exit 0
    fi
done