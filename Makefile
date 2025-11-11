# Makefile for Jupyter Book

# Find all markdown files
MARKDOWN_FILES = $(wildcard markdown/*.md)
# Create corresponding notebook file names
NOTEBOOK_FILES = $(patsubst markdown/%.md,notebooks/%.ipynb,$(MARKDOWN_FILES))
# Create corresponding executed notebook file names
EXECUTED_NOTEBOOK_FILES = $(patsubst notebooks/%.ipynb,notebooks_executed/%.ipynb,$(NOTEBOOK_FILES))

.PHONY: all notebook execute clean

all: notebook

# Convert markdown to notebook for interactive use
notebook: $(NOTEBOOK_FILES)

notebooks/%.ipynb: markdown/%.md
	mkdir -p notebooks
	jupytext --to notebook $< -o $@

# Execute notebooks
execute: $(EXECUTED_NOTEBOOK_FILES)

notebooks_executed/%.ipynb: notebooks/%.ipynb
	mkdir -p notebooks_executed
	jupyter nbconvert --to notebook --execute $< --output-dir notebooks_executed/
