import sys, json, argparse
sys.path.append('YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE/pipeline')
import os
import argparse
from multiagent_platform import EXECUTE_TOOL_CHAIN
from openai import OpenAI


client = OpenAI(api_key="your-deepseek-api-key", base_url="https://api.deepseek.com")

INSTRUCTION = f"""
You are a football expert. You are provided with a question 'Q' and four options 'O1', 'O2', 'O3', and 'O4'.
Before I have used a helpful soccer multi-agent system to solve this process, I will tell you the total process of how agent deal with this problem. 
Please answer the question with one option that best matches the question (replay with 'O1', 'O2', 'O3', or 'O4'). 
Do not include any other text or explanations!!!
"""

def workflow(input_text, Instruction=INSTRUCTION, follow_up_prompt=None, max_tokens_followup=1500):
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text}
        ],
    )
    first_round_reply = completion.choices[0].message.content
    return first_round_reply

import re

def process_football_question(input_dict):
    if "openA_process" in input_dict and "answer" in input_dict:
        return input_dict

    question = input_dict.get("Q", "")
    materials = input_dict.get("materials", "")

    options = {key: value for key, value in input_dict.items() if key.startswith("O")}
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])

    openA_process = EXECUTE_TOOL_CHAIN(question, materials, options_str)

    prompt = f"""
This football question is "{question}". The four corresponding options are:
{options_str}

The processing through the multi-agent platform is as follows:
{openA_process}

Please provide your answer:
"""

    processed_prompt = workflow(prompt)

    answer_match = re.search(r"O\d+", processed_prompt)
    answer = answer_match.group(0) if answer_match else None

    result_dict = input_dict.copy()
    result_dict["openA_process"] = openA_process
    result_dict["answer"] = answer

    return result_dict


from tqdm import tqdm
import json

def process_json_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        progress_bar = tqdm(data_list, desc="Processing (Accuracy: N/A)", unit="item")

        for i, item in enumerate(progress_bar):
            try:
                updated_item = process_football_question(item)
                data_list[i] = updated_item

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=4)

            except ValueError as ve:
                print(f"ValueError processing item {i}: {ve}")
                continue

            except Exception as e:
                print(f"Unexpected error processing item {i}: {e}")
                continue

            correct_count = 0
            total_count = 0
            for entry in data_list:
                if "openA_process" in entry and "answer" in entry:
                    total_count += 1
                    if entry["answer"] == entry.get("closeA"):
                        correct_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0

            progress_bar.set_description(f"Processing (Accuracy: {accuracy:.2%})")
            progress_bar.refresh() 

        print(f"Processing completed. Output saved to {output_file}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file containing football questions.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file. See https://huggingface.co/datasets/Homie0609/SoccerBench/raw/main/qa/q1.json as an example")
    parser.add_argument("--output_file", type=str, help="Path to save the output JSON file. You can just set an json path.")

    args = parser.parse_args()
    process_json_file(args.input_file, args.output_file)


