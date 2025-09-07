from openai import OpenAI
import json
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

    
def extract_entity_info(question):
    """
    Extracts the type and exact name of a football-related entity (player, referee, team, venue) from a given question.

    Args:
        question (str): The question containing the entity.

    Returns:
        tuple: A tuple containing the type and name of the entity, e.g., ("player", "Lionel Messi").
    """
    # Define the instruction
    Instruction = """
    You are an intelligent assistant that can analyze questions related to football. Your task is to identify the type of entity mentioned in the question and extract the exact name of the entity. The entity types are: player, referee, team, venue. If the entity is a coach, classify it as a player. The name extracted should match exactly as it appears in the question.

    Output the result strictly as a tuple in the format: (type, name). Do not include any additional explanations, notes, or formatting.

    For example:
    - Question: "How many goals did Lionel Messi score last season?"
      Output: ("player", "Lionel Messi")
    - Question: "Where is the Camp Nou stadium located?"
      Output: ("venue", "Camp Nou")
    - Question: "What was the decision made by referee Michael Oliver in the last match?"
      Output: ("referee", "Michael Oliver")
    - Question: "How did Manchester United perform in the last game?"
      Output: ("team", "Manchester United")

    However, if the entity type and entity name cannot be determined, please output as: ("unknown", "unknown")

    For example:
    - Question: "Explain the 4-4-2 formation."
      Output: ("unknown", "unknown")
    - Question: "Who is the player in this image?"
      Output: ("player", "unknown")
    """

    # Call the workflow function
    result = workflow(question, Instruction)

    # Crop the output to ensure it's a tuple
    if isinstance(result, str):
        # Remove any unwanted characters or formatting
        result = result.strip().replace('```', '').replace('json', '').strip()
        if result.startswith('(') and result.endswith(')'):
            try:
                # Safely evaluate the string to a tuple
                import ast
                result = ast.literal_eval(result)
            except (ValueError, SyntaxError):
                # Fallback in case of unexpected format
                result = ("unknown", "unknown")
        else:
            # Fallback if the output is not a tuple
            result = ("unknown", "unknown")
    elif not isinstance(result, tuple):
        # Fallback if the output is not a tuple
        result = ("unknown", "unknown")

    return result

import os
import json

def find_json_path(base_folder, entity_type, entity_name):
    """
    Searches for a JSON file in the appropriate subfolder that matches the entity name.
    If no exact match is found, uses the workflow function to determine the best match.

    Args:
        base_folder (str): The base folder containing subfolders (player, referee, team, venue).
        entity_type (str): The type of entity (player, referee, team, venue).
        entity_name (str): The name of the entity to search for.

    Returns:
        str: The absolute path of the matching JSON file.
    """
    # Map entity types to their corresponding subfolder and JSON key
    type_to_key = {
        "player": "FULL_NAME",
        "referee": "DETECTED_NAME",
        "team": "TEAM",
        "venue": "VENUE"
    }

    # Get the subfolder and key based on the entity type
    subfolder = os.path.join(base_folder, entity_type)
    key = type_to_key[entity_type]

    # List to store potential matches
    potential_matches = []

    # Traverse the subfolder and search for matching JSON files
    for root, _, files in os.walk(subfolder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if key in data and data[key] == entity_name:
                        return file_path  # Exact match found
                    elif key in data:
                        potential_matches.append((data[key], file_path))

    # If no exact match is found, use the workflow function to determine the best match
    if potential_matches:
        # Create a prompt with all potential names
        prompt = f"I want to find some information of {entity_type} {entity_name} from my soccer database. Now I have the following possible options\n\n"
        for name, _ in potential_matches:
            prompt += f"- {name}\n"

        prompt += "\nPlease help me determine the best match in the candidate list that is the most likely to be the entity I want. You can just reply me with the exact name of the cantidate you think is the best match without any other words.Please think it carefully and don't give me the wrong answer. I trust you. If really none of them are possible, return me with 'No Matching'\n"
        # print(prompt)
        # Use the workflow function to determine the best match
        best_match = workflow(prompt, "You are an assistant of entity search in soccer database. Determine the best match for the given entity name.")
        # print(best_match)
        # Find the file path corresponding to the best match
        for name, file_path in potential_matches:
            if name == best_match:
                return file_path

    # If no match is found at all, return None
    return "No matching file found."

def TEXTUAL_ENTITY_SEARCH(question, material=None, base_folder = os.path.join(PROJECT_PATH, "database/SoccerWiki/data")):
    entity_type, entity_name = extract_entity_info(question)
    if entity_type == "unknown" or entity_name == "unknown":
        return "No matching file found."
    else:
        json_path = find_json_path(base_folder, entity_type, entity_name)
        return f"The wiki information of this entity could be found in {json_path}."
    
