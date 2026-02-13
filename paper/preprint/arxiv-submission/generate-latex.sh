#!/bin/bash
# Script to generate LaTeX file for arXiv submission
# This uses the same Docker command as the GitHub workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Generating LaTeX file for arXiv submission..."
echo "Working directory: $REPO_ROOT"

docker run --rm --platform linux/amd64 \
  -v "$REPO_ROOT:/data" \
  -w /data \
  pandoc/latex:latest \
  -s paper/preprint/arxiv.md \
  --from=markdown \
  --pdf-engine=xelatex \
  --citeproc \
  --resource-path=paper \
  -o paper/preprint/arxiv-submission/arxiv.tex

# Fix figure paths - Pandoc might generate paths like "paper/figures/figure1.png"
# but arXiv needs relative paths like "figures/figure1.png"
echo "Fixing figure paths..."
sed -i.bak 's|paper/figures/|figures/|g' "$REPO_ROOT/paper/preprint/arxiv-submission/arxiv.tex"
rm -f "$REPO_ROOT/paper/preprint/arxiv-submission/arxiv.tex.bak"

echo ""
echo "✓ LaTeX file generated: paper/preprint/arxiv-submission/arxiv.tex"
echo "✓ Figure paths fixed to be relative (figures/figureX.png)"
echo ""
echo "Next steps:"
echo "1. Review the generated arxiv.tex file"
echo "2. Test compilation:"
echo "   cd paper/preprint/arxiv-submission"
echo "   xelatex arxiv.tex"
echo "   bibtex arxiv"
echo "   xelatex arxiv.tex"
echo "   xelatex arxiv.tex"
echo "3. Zip the entire arxiv-submission folder for arXiv upload:"
echo "   cd paper/preprint"
echo "   zip -r arxiv-submission.zip arxiv-submission/"
