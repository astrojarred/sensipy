# arXiv Submission Package

This folder contains all files needed for arXiv submission.

## Files Included

- `arxiv.tex` - Main LaTeX source file (generate using script below)
- `paper.bib` - Bibliography file
- `figures/` - Directory containing all figures referenced in the paper
  - `figure1.png`
  - `figure2.png`
  - `figure3.png`

## Generating the LaTeX File

Run the generation script:

```bash
./generate-latex.sh
```

Or manually using Docker (same as GitHub workflow):

```bash
cd /path/to/sensipy
docker run --rm --platform linux/amd64 \
  -v "$(pwd):/data" \
  -w /data \
  pandoc/latex:latest \
  -s paper/preprint/arxiv.md \
  --from=markdown \
  --pdf-engine=xelatex \
  --citeproc \
  --resource-path=paper \
  -o paper/preprint/arxiv-submission/arxiv.tex
```

## Testing Compilation

Before submitting to arXiv, test that the LaTeX compiles:

```bash
cd paper/preprint/arxiv-submission
pdflatex arxiv.tex
bibtex arxiv
pdflatex arxiv.tex
pdflatex arxiv.tex
```

Or with XeLaTeX (if you used xelatex in pandoc):

```bash
xelatex arxiv.tex
bibtex arxiv
xelatex arxiv.tex
xelatex arxiv.tex
```

## arXiv Submission

1. Ensure `arxiv.tex` compiles successfully
2. Verify all figures are in the `figures/` directory
3. Create a clean zip file with only essential files:
   ```bash
   cd paper/preprint/arxiv-submission
   ./create-submission-zip.sh
   ```
   This creates `arxiv-submission.zip` in `paper/preprint/` with only:
   - `arxiv.tex` (main LaTeX file)
   - `paper.bib` (bibliography)
   - `figures/` directory (all figures)
   
   Helper files (README.md, CHECKLIST.md, scripts, etc.) are automatically excluded.
4. Upload `arxiv-submission.zip` to arXiv

## Notes

- arXiv will auto-detect the main `.tex` file (usually the one with `\documentclass`)
- All figure paths in the LaTeX should be relative (e.g., `figures/figure1.png`)
- Make sure the bibliography file is named correctly (`paper.bib`)
