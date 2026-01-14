#!/bin/bash

# ============================================================================
# Git Commit and Push Script
# ============================================================================
# This script helps you commit and push changes to GitHub in one go
# ============================================================================
# Shows what will be committed
# Prompts before staging changes
# Asks for commit message
# Confirms before pushing
# Usage:
# ./git_push.sh                    # Interactive mode (prompts for message)
# ./git_push.sh "Your commit msg"   # With commit message

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository!"
    exit 1
fi

# Get the current branch
CURRENT_BRANCH=$(git branch --show-current)
print_info "Current branch: $CURRENT_BRANCH"

# Check if there are any changes
if git diff --quiet && git diff --cached --quiet; then
    print_warning "No changes to commit. Working tree is clean."
    exit 0
fi

# Show status
echo ""
print_info "Current status:"
git status --short

# Show what will be committed
echo ""
print_info "Files to be committed:"
if git diff --cached --quiet; then
    print_warning "No files staged. Staging all changes..."
    git add .
else
    print_info "Staged files:"
    git diff --cached --name-only
    echo ""
    read -p "Stage all remaining changes? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        print_success "All changes staged"
    fi
fi

# Get commit message
echo ""
if [ -z "$1" ]; then
    print_info "Enter commit message (or press Enter for default):"
    read -r COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
        print_info "Using default commit message: $COMMIT_MSG"
    fi
else
    COMMIT_MSG="$1"
    print_info "Using provided commit message: $COMMIT_MSG"
fi

# Commit changes
echo ""
print_info "Committing changes..."
if git commit -m "$COMMIT_MSG"; then
    print_success "Changes committed successfully"
else
    print_error "Commit failed!"
    exit 1
fi

# Show commit summary
echo ""
print_info "Commit summary:"
git log -1 --stat

# Ask if user wants to push
echo ""
read -p "Push to remote? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Skipping push. You can push later with: git push"
    exit 0
fi

# Push to remote
echo ""
print_info "Pushing to origin/$CURRENT_BRANCH..."
if git push -u origin "$CURRENT_BRANCH"; then
    print_success "Successfully pushed to GitHub!"
    echo ""
    print_info "Repository: https://github.com/GabenS99/manarat7_benchmark"
else
    print_error "Push failed!"
    exit 1
fi

print_success "All done! ✓"
