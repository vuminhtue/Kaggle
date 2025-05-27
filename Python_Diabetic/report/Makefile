# Makefile for compiling the diabetic data analysis LaTeX report

# Define variables
MAIN = main
TEX = $(MAIN).tex
PDF = $(MAIN).pdf

# Default target
all: $(PDF)

# Rule to make the PDF using pdflatex
$(PDF): $(TEX) chapters/*.tex
	pdflatex $(TEX)
	pdflatex $(TEX)  # Run twice for references and TOC

# Clean up auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.nav *.snm *.vrb *.dvi *.fls *.fdb_latexmk *.synctex.gz

# Clean up everything including the PDF
distclean: clean
	rm -f $(PDF)

# Phony targets
.PHONY: all clean distclean 