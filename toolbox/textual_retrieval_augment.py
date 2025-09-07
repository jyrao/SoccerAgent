from openai import OpenAI
import json, os
from tqdm import tqdm
import argparse

######################## Parameters ########################
PROJECT_PATH = "YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE"
client = OpenAI(api_key="your-deepseek-api-key", base_url="https://api.deepseek.com")

def workflow(input_text, Instruction, follow_up_prompt=None, max_tokens_followup=1500):

    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text}
        ],
        stream=False 
    )
    
    first_round_reply = completion.choices[0].message.content
    
    if follow_up_prompt:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": Instruction},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": first_round_reply},
                {"role": "user", "content": follow_up_prompt}
            ],
            max_tokens=max_tokens_followup,
            stream=False
        )
        second_round_reply = completion.choices[0].message.content
        return first_round_reply, second_round_reply
    else:
        return first_round_reply


def generate_textual_RAG_prompt(question, textual_material):
    """
    Generates a prompt based on the question and textual material.

    Args:
        question (str): The user's question.
        textual_material (str): The textual material (JSON file path or raw text).

    Returns:
        str: The generated prompt.
    """
    # Check if textual_material is a JSON file path
    if isinstance(textual_material, str) and textual_material.endswith(".json"):
        try:
            # Read the JSON file and convert it to a formatted string
            with open(textual_material, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                formatted_json = json.dumps(json_data, indent=4, ensure_ascii=False)
                textual_material = formatted_json
        except Exception as e:
            return f"Failed to read JSON file: {e}"

    # Generate the prompt
    prompt = f"""
    Question: {question}
    Contextual Material: {textual_material}
    Please answer the question based on the provided contextual material.
    """
    return prompt

def TEXTUAL_RETRIEVAL_AUGMENT(question, textual_material):
    """
    Retrieves and augments textual material to answer the question using the workflow function.

    Args:
        question (str): The user's question.
        textual_material (str): The textual material (JSON file path or raw text).

    Returns:
        str: The generated answer.
    """
    # Define the instruction for the agent
    Instruction = "You are an assistant that answers questions based on provided contextual material."
    file_path = os.path.join(PROJECT_PATH, textual_material[0])

    # Generate the prompt
    prompt = generate_textual_RAG_prompt(question, file_path)
    answer = None
    try:
    # Call the workflow function to generate the answer
        answer = workflow(prompt, Instruction)
    except:
        answer = "Failed in LLM Generation."
    # Return the QA output
    return answer