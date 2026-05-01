#!/bin/bash

# LaTeX Compilation Script with Preview Support
# Usage: ./compile_latex.sh <tex_file> [--preview]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if tex file is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: No .tex file specified${NC}"
    echo "Usage: $0 <tex_file> [--preview]"
    exit 1
fi

TEX_FILE="$1"
PREVIEW=false

# Check for preview flag
if [ "$2" == "--preview" ]; then
    PREVIEW=true
fi

# Check if file exists
if [ ! -f "$TEX_FILE" ]; then
    echo -e "${RED}Error: File '$TEX_FILE' not found${NC}"
    exit 1
fi

# Get directory and filename without extension
DIR=$(dirname "$TEX_FILE")
BASENAME=$(basename "$TEX_FILE" .tex)

echo -e "${GREEN}=== LaTeX Compilation ===${NC}"
echo "File: $TEX_FILE"
echo "Directory: $DIR"
echo ""

# Change to the document directory
cd "$DIR"

echo -e "${YELLOW}[1/4] First pdflatex pass...${NC}"
pdflatex -interaction=nonstopmode "$BASENAME.tex" > /dev/null 2>&1 || {
    echo -e "${RED}First pdflatex pass failed. Running with output:${NC}"
    pdflatex -interaction=nonstopmode "$BASENAME.tex"
    exit 1
}

echo -e "${YELLOW}[2/4] Running bibtex...${NC}"
bibtex "$BASENAME" > /dev/null 2>&1 || {
    echo -e "${YELLOW}Warning: bibtex had warnings (this is often normal)${NC}"
}

echo -e "${YELLOW}[3/4] Second pdflatex pass...${NC}"
pdflatex -interaction=nonstopmode "$BASENAME.tex" > /dev/null 2>&1 || {
    echo -e "${RED}Second pdflatex pass failed.${NC}"
    exit 1
}

echo -e "${YELLOW}[4/4] Third pdflatex pass...${NC}"
pdflatex -interaction=nonstopmode "$BASENAME.tex" > /dev/null 2>&1 || {
    echo -e "${RED}Third pdflatex pass failed.${NC}"
    exit 1
}

# Clean up auxiliary files
echo -e "${GREEN}Cleaning up auxiliary files...${NC}"
rm -f *.aux *.log *.bbl *.blg *.toc *.out *.lof *.lot 2>/dev/null || true

PDF_FILE="${BASENAME}.pdf"

if [ -f "$PDF_FILE" ]; then
    echo -e "${GREEN}✓ Compilation successful!${NC}"
    echo -e "${GREEN}Output: $DIR/$PDF_FILE${NC}"

    # Show file size
    SIZE=$(ls -lh "$PDF_FILE" | awk '{print $5}')
    echo -e "Size: $SIZE"

    # Preview if requested
    if [ "$PREVIEW" = true ]; then
        echo -e "${GREEN}Opening preview...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open "$PDF_FILE"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            xdg-open "$PDF_FILE" 2>/dev/null || evince "$PDF_FILE" 2>/dev/null || echo "Please install a PDF viewer"
        else
            echo "Preview not supported on this OS. Please open $PDF_FILE manually."
        fi
    fi
else
    echo -e "${RED}✗ Compilation failed - PDF not generated${NC}"
    exit 1
fi

echo -e "${GREEN}=== Done ===${NC}"
