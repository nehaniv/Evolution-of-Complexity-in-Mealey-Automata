# Evolution-of-Complexity-in-Mealey-Automata

# Mealy Automata Evolution Framework

This project implements an **evolutionary strategy (ES)** for evolving **Mealy automata** to solve predictive tasks such as:
- Traversal of deterministic environments (e.g., "SimpleHardestEnvironment")
- Pattern prediction (e.g., EightBall, FogelPalindrome)

All logic and functionality are contained in a single Python file:
- `ES-Automata-Fogel.py`

A bash script is provided for running the system with customizable parameters:
- `MealeyEvolutionStrategy.sh`

##  Environment Setup

We recommend using a Python virtual environment to manage dependencies cleanly.

### Prerequisites
- Python 3.10 or newer (tested with Python 3.13)
- [GAP](https://www.gap-system.org/) (for algebraic decomposition of automata)
- `graphviz' (for visualization of automata) 
- `pdflatex` (for LaTeX report generation)

### Quick Setup Using `venv`

```bash
# Create and activate virtual environment
python3 -m venv automata-env
source automata-env/bin/activate  # or 'automata-env\Scripts\activate' on Windows

# Install dependencies
pip install -r requirements.txt



#**## Option 2: Conda**


conda env create -f environment.yml
conda activate automata-env



###  Running the Evolution
bash MealeyEvolutionStrategy.sh


You can modify MealeyEvolutionStrategy.sh to adjust:

Population size

Number of states

Number of offspring

Fitness function

Initialization method (e.g. self-looped, random, from file)

Environment variant

All results are timestamped and saved:

LaTeX reports (.tex, compiled to .pdf)

GAP scripts and outputs (.g, .txt)

Graphs (.pdf)

Summary Excel files (.xlsx)

Pickled best automata (.pkl)
