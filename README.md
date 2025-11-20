# AI Course Repository

This repository contains the materials for an AI course, focusing on object detection, model finetuning, and evaluation. The content is created using Jupyter notebooks generated from markdown files.

## Prerequisites

Before you begin, ensure you have Python installed. All the necessary Python packages are listed in the `requirements.txt` file.

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the data by running the script in the `data` directory:
    ```bash
    bash data/get_data.sh
    ```

## Usage

This repository uses a `Makefile` to automate the workflow of generating and executing notebooks.

### Main Commands

*   **`make all`**
    This is the primary command that runs the entire pipeline: generates notebooks, executes them, and converts them to HTML.

*   **`make notebook`**
    Converts the markdown files in the `markdown/` directory into Jupyter notebooks in the `notebooks/` directory. This is useful if you want to interact with the notebooks directly.

*   **`make execute`**
    Executes the Jupyter notebooks located in the `notebooks/` directory using Papermill. The executed notebooks, with all the cell outputs, are saved in the `notebooks_executed/` directory.

*   **`make html`**
    Converts the executed notebooks from `notebooks_executed/` into HTML files, which are saved in the `notebooks_executed_html/` directory for easy viewing in a web browser.

*   **`make clean`**
    Removes all generated files, including notebooks, executed notebooks, HTML files, and other artifacts.

## Workflow

The intended workflow is as follows:

1.  Modify the content by editing the markdown files in the `markdown/` directory. These are the source files.
2.  Run `make all` to regenerate the notebooks, execute them, and create the HTML reports.

If you wish to experiment or work interactively, you can run `make notebook` and then open the generated notebooks in the `notebooks/` directory using Jupyter. However, be aware that any changes made directly to the notebooks in `notebooks/` will be overwritten the next time `make notebook` or `make all` is run.

## Directory Structure

-   `markdown/`: Source files for the notebooks in Markdown format.
-   `notebooks/`: Jupyter notebooks generated from the markdown files for interactive use.
-   `notebooks_executed/`: Executed notebooks with cell outputs.
-   `notebooks_executed_html/`: HTML versions of the executed notebooks.
-   `data/`: Contains datasets and scripts to download them.
-   `requirements.txt`: A list of Python packages required for this project.
-   `Makefile`: Defines the commands for the project workflow.
