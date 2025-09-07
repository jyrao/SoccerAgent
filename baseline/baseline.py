import os
import pandas as pd
import argparse
from model import QwenVL, GPT4o


def main():
    # Setting up command-line arguments
    parser = argparse.ArgumentParser(description="Run QA tests with different models.")
    parser.add_argument('--model', choices=['qwen', 'gpt'], required=True, help='The model to use (qwen or gpt)')
    parser.add_argument('--input_file', required=True, help='The input JSON file for the QA task')
    parser.add_argument('--output_file', required=True, help='The output CSV file to store the results')
    parser.add_argument('--api_key', default=None, help='API key for GPT model (if applicable)')
    parser.add_argument('--model_path', default=None, help='Model path for Qwen model (if applicable)')
    args = parser.parse_args()

    # Handle Qwen model
    if args.model == 'qwen':
        if args.model_path is None:
            raise ValueError("Model path must be provided for Qwen model.")
        agent = QwenVL(model_path=args.model_path)
    # Handle GPT4o model
    elif args.model == 'gpt':
        if args.api_key is None:
            raise ValueError("API key must be provided for GPT model.")
        agent = GPT4o(api_key=args.api_key)

    # Create the output file if it doesn't exist
    if not os.path.exists(args.output_file):
        df = pd.DataFrame(columns=['answer', 'option_num'])
        df.to_csv(args.output_file, index=False)

    # Running the QA test with the given model and file
    agent.test_qa(args.input_file, args.output_file)
    accuracy = agent.cal_acc(args.output_file, args.input_file)

    print(f"Accuracy of {args.model} model: {accuracy:.2f}%, results saved to {args.output_file}")


if __name__ == "__main__":
    main()
