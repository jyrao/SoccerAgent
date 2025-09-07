#### GAME INFO RETRIEVAL & MATCH HISTORY RETRIEVAL

import json
from openai import OpenAI
import json
from tqdm import tqdm
import argparse, os

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



def generate_commentary_from_json_matchtime(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    event_list = data.get("annotations", [])
    if not event_list:
        return "No annotations found in the JSON file."
    
    result = []
    
    for event in reversed(event_list): 
        timestamp = event.get("contrastive_aligned_gameTime", "")
        if not timestamp:
            timestamp = event.get("gameTime", "")
        if not timestamp:
            continue  
        
        try:
            half, time = timestamp.split(" - ")
            if half == "1":
                half_str = "1st half"
            elif half == "2":
                half_str = "2nd half"
            else:
                continue 
        except ValueError:
            continue 
        
        description = event.get("description", "")
        if not description:
            continue
        
        commentary_line = f"{half_str} - {time} \"{description}\""
        result.append(commentary_line)
    
    return "\n".join(result)

def generate_commentary_from_json_1988(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    comments_list = data.get("comments", [])
    if not comments_list:
        return "No comments found in the JSON file."
    
    result = []
    
    for comment in comments_list:
        half = comment.get("half")
        if half not in [1, 2]:
            continue 
        
        timestamp = comment.get("time_stamp", "")
        if not timestamp:
            continue 
        
        comments_text = comment.get("comments_text", "")
        if not comments_text:
            continue 
        
        half_str = "1st half" if half == 1 else "2nd half"
        
        commentary_line = f"{half_str} - {timestamp} \"{comments_text}\""
        result.append(commentary_line)
    
    return "\n".join(result)

def get_match_info_matchtime(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if "annotations" in data:
        del data["annotations"]
    
    result = json.dumps(data, indent=4, ensure_ascii=False)
    
    return result

def get_match_info_1988(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if "comments" in data:
        del data["comments"]
    
    result = json.dumps(data, indent=4, ensure_ascii=False)
    
    return result

def get_match_info(json_file_path):
    basename = os.path.basename(json_file_path)
    
    if basename == "Labels-caption.json":
        return get_match_info_matchtime(json_file_path)
    else:
        return get_match_info_1988(json_file_path)
    
def generate_commentary_from_json(json_file_path):
    basename = os.path.basename(json_file_path)
    
    if basename == "Labels-caption.json":
        return generate_commentary_from_json_matchtime(json_file_path)
    else:
        return generate_commentary_from_json_1988(json_file_path)
    
def MATCH_HISTORY_RETRIEVAL(query, material):
    if len(material) == 0:
        return "Something went wrong with this question."
    
    file_path = os.path.join(PROJECT_PATH, material[0])
    match_history = generate_commentary_from_json(file_path)
    prompt = f"""Here is a question about soccer game: 
    
    "{query}"

The match history information has been found as following shows, you need to answer the question based on the information provided:

    {match_history}

Please provide the answer based on the match history information. Please think it carefully and make sure your answer is evidence-based and accurate. Now answer the question in the following format:

[ANSWER]: [Your answer here]
[EXPLANATION & REASONING]: [Your explanation here]

You should return exactly in this form without any other words.
    """

    answer = workflow(prompt, "You are a soccer expert that answers questions based on match history information.")

    return answer

def GAME_INFO_RETRIEVAL(query, material):
    if len(material) == 0:
        return "Something went wrong with this question."
    
    file_path = os.path.join(PROJECT_PATH, material[0])
    match_info = get_match_info(file_path)
    prompt = f"""Here is a question about soccer game: 
    
    "{query}"

The match related information has been found as following shows, you need to answer the question based on the information provided:

    {match_info}

Please provide the answer based on the match related information. Please think it carefully and make sure your answer is evidence-based and accurate. Now answer the question in the following format:

[ANSWER]: [Your answer here]
[EXPLANATION & REASONING]: [Your explanation here]

You should return exactly in this form without any other words.
    """

    answer = workflow(prompt, "You are a soccer expert that answers questions based on match history information.")

    return answer
