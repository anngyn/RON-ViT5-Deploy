#!/bin/bash
set -e

echo "=== Push RON-ViT5-Deploy to GitHub ==="
echo ""

# Replace with your repo URL
REPO_URL="https://github.com/anngyn/RON-ViT5-Deploy.git"

# Check git installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git not installed"
    echo "Install: sudo apt install git"
    exit 1
fi

echo "Current directory: $(pwd)"
echo ""

# Initialize git if needed
if [ ! -d .git ]; then
    echo "Initializing git repo..."
    git init
fi

echo ""
echo "Staging files..."
git add .

echo ""
echo "Creating commit..."
git commit -m "Initial commit: RON-ViT5 training pipeline

- 2 training scripts (baseline, noisy_aug)
- Core modules (dataset, models, noise, train, evaluate)
- 7 configs including 16GB GPU variants
- 5 documentation guides
- Setup and run scripts"

echo ""
echo "Setting remote..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

echo ""
echo "Renaming branch to main..."
git branch -M main

echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "=== Push complete! ==="
echo ""
echo "Next steps:"
echo "1. Go to: https://github.com/anngyn/RON-ViT5-Deploy"
echo "2. Verify all files uploaded"
echo "3. Update QUICKSTART.md with real repo URL"
