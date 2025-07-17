import argparse
import pickle
import sys
# If your file is named ES-Automata-Fogel.py, Python cannot import it due to the hyphen. Rename it to ES_Automata_Fogel.py if needed, then:
from ES_Automata_Fogel import create_gap_report, create_summary_excel, create_report, parse_gap_output


def main():
    parser = argparse.ArgumentParser(description="Post-GAP reporting for Mealy Automata Evolution")
    parser.add_argument('--stamp', type=str, required=True)
    parser.add_argument('--fitness', type=str, required=True)
    parser.add_argument('--env_variant', type=str, required=True)
    args = parser.parse_args()

    # Load best_per_run and params from the checkpoint
    checkpoint_path = f'checkpoint_{args.stamp}.pkl'
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Please run the main evolution script first to generate it.")
        sys.exit(1)
    best_per_run = checkpoint_data['best_per_run']
    params = checkpoint_data['params']

    gap_output_file = f"gap_output_{args.stamp}.txt"
    create_gap_report(gap_output_file, args.stamp, args.fitness, args.env_variant, params)
    create_summary_excel(best_per_run, parse_gap_output(gap_output_file), args.stamp)
    parsed_results = parse_gap_output(gap_output_file)
    create_report(args.fitness, args.env_variant, args.stamp, best_per_run, parsed_results, params)

if __name__ == "__main__":
    main()


