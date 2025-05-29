#!/bin/bash

##########################################
# Mealey Automata Evolution Strategy Runner
#
# This script runs ES-Automata-Fogel.py with various modes:
# - Default: Create a new population (random or self-looping)
# - Load Automaton: Initialize population from a saved automaton file
# - Load Population: Initialize from a saved population file
# - Resume: Resume from a checkpoint
#
# Usage:
# ./MealeyEvolutionStrategy.sh [OPTIONS]
#
# Example:
# ./MealeyEvolutionStrategy.sh --fitness Traversal --env_variant SimpleHardestEnvironment --init_automaton_file my_automaton.pkl
#
##########################################

# -------- Default Parameters --------
POP_SIZE=5
OFFSPRING_SIZE=50
NUM_STATES=20
NUM_RUNS=10
GENERATIONS=50000
FITNESS="Traversal"  # Options: EightBall, FogelPalindrome, Traversal
#FITNESS="FogelPalindrome"  # Options: EightBall, FogelPalindrome, Traversal
#ENV_VARIANT="NA "
ENV_VARIANT="SimpleHardestEnvironment"  # For Traversal fitness
#ENV_VARIANT="SimpleEasiestEnvironment"  # For Traversal fitness
SELF_LOOP_INIT="True"
INIT_AUTOMATON_FILE=""
INIT_POPULATION_FILE=""
CHECKPOINT_FILE=""
HELP=""

# -------- Help Function --------
show_help() {
    echo "Usage: ./MealeyEvolutionStrategy.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --population_size N         Set the population size (default: 20)"
    echo "  --offspring_size N          Set the offspring size (default: 100)"
    echo "  --num_states N              Number of states per automaton (default: 50)"
    echo "  --runs N                    Number of runs (default: 10)"
    echo "  --generations N             Number of generations (default: 1000)"
    echo "  --fitness NAME              Fitness function: EightBall, FogelPalindrome, Traversal"
    echo "  --env_variant NAME          Traversal variant: SimpleHardestEnvironment or SimpleEasiestEnvironment"
    echo "  --self_loop_init True/False Initialize automata with self-looping transitions (default: True)"
    echo "  --init_automaton_file PATH  Load automaton from pickle file to initialize population"
    echo "  --init_population_file PATH Load population from pickle file"
    echo "  --checkpoint_file PATH      Resume from a checkpoint file"
    echo "  --help                      Show this help message"
    exit 0
}

# -------- Parse Command-Line Arguments --------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --population_size) POP_SIZE="$2"; shift ;;
        --offspring_size) OFFSPRING_SIZE="$2"; shift ;;
        --num_states) NUM_STATES="$2"; shift ;;
        --runs) NUM_RUNS="$2"; shift ;;
        --generations) GENERATIONS="$2"; shift ;;
        --fitness) FITNESS="$2"; shift ;;
        --env_variant) ENV_VARIANT="$2"; shift ;;
        --self_loop_init) SELF_LOOP_INIT="$2"; shift ;;
        --init_automaton_file) INIT_AUTOMATON_FILE="$2"; shift ;;
        --init_population_file) INIT_POPULATION_FILE="$2"; shift ;;
        --checkpoint_file) CHECKPOINT_FILE="$2"; shift ;;
        --help) show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# -------- Generate Timestamp --------
STAMP=$(date +"%Y%m%d_%H%M%S")

# -------- Show What Will Run --------
echo "=========================================="
echo " Running ES-Automata-Fogel.py with:"
echo " Population size    : $POP_SIZE"
echo " Offspring size     : $OFFSPRING_SIZE"
echo " Num states         : $NUM_STATES"
echo " Runs               : $NUM_RUNS"
echo " Generations        : $GENERATIONS"
echo " Fitness            : $FITNESS"
echo " Env Variant        : $ENV_VARIANT"
echo " Self Loop Init     : $SELF_LOOP_INIT"
if [ -n "$INIT_AUTOMATON_FILE" ]; then
    echo " Init Automaton     : $INIT_AUTOMATON_FILE"
fi
if [ -n "$INIT_POPULATION_FILE" ]; then
    echo " Init Population    : $INIT_POPULATION_FILE"
fi
if [ -n "$CHECKPOINT_FILE" ]; then
    echo " Checkpoint File    : $CHECKPOINT_FILE"
fi
echo "=========================================="

# -------- Build Python Command --------
PYTHON_CMD="python3 ES-Automata-Fogel.py \
  --population_size $POP_SIZE \
  --offspring_size $OFFSPRING_SIZE \
  --num_states $NUM_STATES \
  --runs $NUM_RUNS \
  --generations $GENERATIONS \
  --fitness $FITNESS \
  --env_variant $ENV_VARIANT \
  --self_loop_init $SELF_LOOP_INIT \
  --stamp $STAMP"

# Add optional files
if [ -n "$INIT_AUTOMATON_FILE" ]; then
    PYTHON_CMD+=" --init_automaton_file $INIT_AUTOMATON_FILE"
fi

if [ -n "$INIT_POPULATION_FILE" ]; then
    PYTHON_CMD+=" --init_population_file $INIT_POPULATION_FILE"
fi

if [ -n "$CHECKPOINT_FILE" ]; then
    PYTHON_CMD+=" --checkpoint_file $CHECKPOINT_FILE"
fi

# -------- Run Python Script --------
echo "Running: $PYTHON_CMD"
eval $PYTHON_CMD

# -------- Compile LaTeX Reports --------
for REPORT in evolution_prelim_report_${STAMP}.tex evolution_report_${STAMP}.tex gap_analysis_report_${STAMP}.tex; do
  if [ -f "$REPORT" ]; then
    echo "Compiling LaTeX report: $REPORT"
    pdflatex "$REPORT" > /dev/null
  fi
done

# -------- Open the PDFs --------
for PDF in evolution_report_${STAMP}.pdf gap_analysis_report_${STAMP}.pdf; do
  if [ -f "$PDF" ]; then
    echo "Opening report: $PDF"
    open "$PDF"
  else
    echo "Warning: PDF report $PDF not generated."
  fi
done

