# Makefile for Jupyter Book

# Find all markdown files
MARKDOWN_FILES = $(wildcard markdown/*.md)
# Create corresponding notebook file names
NOTEBOOK_FILES = $(patsubst markdown/%.md,notebooks/%.ipynb,$(MARKDOWN_FILES))
# Create corresponding executed notebook file names
EXECUTED_NOTEBOOK_FILES = $(patsubst notebooks/%.ipynb,notebooks_executed/%.ipynb,$(NOTEBOOK_FILES))
# Create corresponding html files from executed notebooks
HTML_FILES = $(patsubst notebooks_executed/%.ipynb,notebooks_executed_html/%.html,$(EXECUTED_NOTEBOOK_FILES))

.PHONY: all notebook execute html clean compress_notebooks

all: notebook execute html

# Convert markdown to notebook for interactive use
notebook: $(NOTEBOOK_FILES)

notebooks/%.ipynb: markdown/%.md
	mkdir -p notebooks
	jupytext --to notebook $< -o $@

# Execute notebooks
execute: $(EXECUTED_NOTEBOOK_FILES)

notebooks_executed/module4_pt3_finetuning.ipynb: notebooks_executed/module4_pt2_open_set_detection.ipynb
notebooks_executed/module4_pt4_evaluation.ipynb: notebooks_executed/module4_pt3_finetuning.ipynb
notebooks_executed/module4_pt5_sam.ipynb: notebooks_executed/module4_pt3_finetuning.ipynb

notebooks_executed/%.ipynb: notebooks/%.ipynb
	mkdir -p notebooks_executed
	papermill --log-output --cwd `pwd`/notebooks_executed --kernel python3 $< $@

# Convert executed notebooks to html
html: $(HTML_FILES)

notebooks_executed_html/%.html: notebooks_executed/%.ipynb
	mkdir -p $(dir $@)
	jupyter nbconvert --to html $< --output-dir $(dir $@) --output $(notdir $@)

# Compress executed notebooks
compress_notebooks:
	(cd notebooks_executed && find . -name '*.ipynb' -print0 | tar --null -czf ../notebooks_executed.tar.gz -T -)

clean:
	rm -Rf yolov8n.pt
	rm -Rf notebooks_executed/*.ipynb
	rm -Rf notebooks_executed/runs
	rm -Rf notebooks_executed/yolo11n.pt
	rm -Rf notebooks_executed/ena24_yolo_dataset.yaml
	rm -Rf notebooks_executed_html
	rm -Rf data/IDLE-OO-Camera-Traps_yolo
