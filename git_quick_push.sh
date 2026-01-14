#!/bin/bash

# ============================================================================
# Quick Git Push Script (Non-interactive)
# ============================================================================
# Usage: ./git_quick_push.sh [commit_message]
# If no message provided, uses timestamp
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if in git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository!${NC}"
    exit 1
fi

# Get commit message
if [ -z "$1" ]; then
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
else
    COMMIT_MSG="$1"
fi

BRANCH=$(git branch --show-current)

echo -e "${BLUE}Branch:${NC} $BRANCH"
echo -e "${BLUE}Staging all changes...${NC}"
git add .

echo -e "${BLUE}Committing:${NC} $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo -e "${BLUE}Pushing to origin/$BRANCH...${NC}"
git push -u origin "$BRANCH"

echo -e "${GREEN}✓ Successfully pushed!${NC}"
