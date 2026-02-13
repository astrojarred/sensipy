#!/bin/bash
# Create a clean zip file for arXiv submission (excludes helper files)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$SCRIPT_DIR"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZIP_NAME="arxiv-submission.zip"
ZIP_PATH="$PARENT_DIR/$ZIP_NAME"

echo "Creating arXiv submission zip..."
echo "Working directory: $SUBMISSION_DIR"

# Clean up any temporary files first
if [ -f "$SUBMISSION_DIR/cleanup.sh" ]; then
    echo "Running cleanup..."
    "$SUBMISSION_DIR/cleanup.sh" > /dev/null 2>&1 || true
fi

# Create zip with only essential files
cd "$SUBMISSION_DIR"

# Remove old zip if it exists
rm -f "$ZIP_PATH"

# Create zip with only the files arXiv needs:
# - arxiv.tex (main LaTeX file)
# - paper.bib (bibliography, if present)
# - figures/ directory (all figures)
zip -r "$ZIP_PATH" \
    arxiv.tex \
    paper.bib \
    figures/ \
    -x "*.aux" -x "*.bbl" -x "*.blg" -x "*.log" -x "*.out" -x "*.pdf" \
    -x "*.synctex.gz" -x "*.toc" -x "*.lof" -x "*.lot" \
    -x "README.md" -x "CHECKLIST.md" -x "*.sh" -x ".DS_Store"

echo ""
echo "✓ Submission zip created: $ZIP_PATH"
echo ""
echo "Files included in zip:"
unzip -l "$ZIP_PATH" | grep -E "\.(tex|bib|png)$|figures/"
echo ""
echo "Zip file size: $(du -h "$ZIP_PATH" | cut -f1)"
echo ""
echo "Ready to upload to arXiv!"
