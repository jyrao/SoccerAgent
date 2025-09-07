import re
from openai import OpenAI

def workflow(input_text, Instruction="You are an expert of soccer referee.", follow_up_prompt=None, api_key="your-deepseek-api-key", max_tokens_followup=1500):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text}
        ],
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
            max_tokens=max_tokens_followup
        )
        second_round_reply = completion.choices[0].message.content
        return first_round_reply, second_round_reply
    else:
        return first_round_reply

def generate_prompt(question):
    """
    Generates a prompt for classifying the input question into one of the 11 categories.
    
    Args:
        question (str): The input question to classify.
    
    Returns:
        str: A formatted prompt for an LLM to classify the question.
    """
    # Define the context and categories
    context = (
        "You are tasked with classifying a question into one of the following 11 categories:\n"
        "1. Offence - Classifies whether the foul occurred during an offensive play.\n"
        "2. Contact - Determines if there was contact between players during the foul.\n"
        "3. Bodypart - Identifies the body part involved in the foul.\n"
        "4. Upper body part - Specifies the exact upper body part involved in the foul.\n"
        "5. Action class - Categorizes the type of foul action.\n"
        "6. Severity - Evaluates the severity of the foul.\n"
        "7. Multiple fouls - Checks if multiple fouls occurred.\n"
        "8. Try to play - Determines if the player attempted to continue playing after the foul.\n"
        "9. Touch ball - Verifies if the offending player touched the ball.\n"
        "10. Handball - Checks if the foul involved handball.\n"
        "11. Handball offence - Determines if the foul was caused by handball.\n\n"
        "Your task is to analyze the given question and return the category it belongs to.\n"
        "The output format must strictly be:\n"
        "[CLASS]: <Category>\n"
        "Where <Category> is one of the 11 categories listed above (e.g., Offence, Contact, etc.).\n"
        "Do not include any additional text or explanations.\n\n"
        "Question: "
    )
    
    # Combine the context with the input question
    prompt = context + question
    return prompt


def extract_category(output):
    """
    Extracts the category from the LLM output using a regular expression.
    
    Args:
        output (str): The LLM output containing the classification.
    
    Returns:
        str: The extracted category.
    """
    match = re.search(r"\[CLASS\]:\s*(\w+)", output)
    if match:
        return match.group(1)
    return "Unknown"


import sys
sys.path.append('YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE/pipeline/toolbox')

from vlm import VLM

def generate_vlm_prompt(question: str, category: str, index: int) -> str:
    """
    Generates a prompt for a Vision-Language Model (VLM) to classify a question into one of the predefined options.
    
    Args:
        question (str): The input question to answer.
        category (str): The category of the question (e.g., "Offence", "Contact", etc.).
        index (int): The index of the video being analyzed.
    
    Returns:
        str: A formatted prompt for the VLM to answer the question.
    """
    # Define the options for each category
    category_options = {
        "Offence": ["Offence", "No offence", "Between"],
        "Contact": ["With", "Without"],
        "Bodypart": ["Upperbody", "Underbody"],
        "Upper body part": ["Arms", "Shoulder"],
        "Action class": ["St. tackling", "Tackling", "Challenge", "Holding", "Elbowing", "High leg", "Pushing", "Dive"],
        "Severity": ["No card", "No card or yellow card", "Yellow card", "Yellow card or red card", "Red card"],
        "Multiple fouls": ["Yes", "No"],
        "Try to play": ["Yes", "No"],
        "Touch ball": ["Yes", "No"],
        "Handball": ["Yes", "No"],
        "Handball offence": ["Yes", "No"]
    }
    
    # Get the options for the given category
    options = category_options.get(category, [])
    
    # Format the options as a readable string
    options_str = ", ".join([f'"{option}"' for option in options])
    
    # Generate the prompt
    prompt = (
        f"You are a professional football referee tasked with analyzing a video and answering the following question: \"{question}\".\n"
        f"The video you are analyzing is the {index}th video related to this question.\n"
        f"Please provide your answer based solely on the content of the video. Your response MUST(!!!) be strictly one of the following options: {options_str}.\n"
        f"Do not include any additional text or explanations. Provide your answer now, without any other words, only one of the above option itself:"
    )
    
    return prompt

import re
from collections import Counter

def FOUL_RECOGNITION(query: str, materials: list) -> str:
    """
    Recognizes the most frequent foul-related answer across multiple videos.
    
    Args:
        query (str): The input question to analyze.
        materials (list): A list of video paths to be analyzed.
    
    Returns:
        str: The most frequent answer among all videos.
    """
    # Step 1: Generate the category for the query using LLM
    llm_prompt_query = query.split("speed.")[-1] if "speed." in query else query
    prompt = generate_prompt(llm_prompt_query)
    llm_output = workflow(prompt)
    category = extract_category(llm_output)
    # print(category)
    # Step 2: Initialize a list to store VLM answers
    vlm_answers = []
    
    # Step 3: Iterate over each video path in materials
    for idx, video_path in enumerate(materials):
        # Generate the VLM prompt
        index = idx + 1  # Convert 0-based index to 1-based index
        vlm_prompt = generate_vlm_prompt(question=llm_prompt_query, category=category, index=index)
        # Simulate calling the VLM with the current video path
        # For demonstration purposes, we assume the `vlm` function exists and works as expected
        vlm_response = VLM(query=vlm_prompt, material=[video_path])
        vlm_answers.append(vlm_response)

    # Step 4: Determine the most frequent answer
    if not vlm_answers:
        return "No answers found"
    
    # Count the frequency of each answer
    answer_counts = Counter(vlm_answers)

    # Find the most frequent answer(s)
    max_frequency = max(answer_counts.values())  # Get the highest frequency
    most_frequent_answers = [ans for ans, count in answer_counts.items() if count == max_frequency]

    # If there's a tie, return the first one that appears in the original list
    most_frequent_answer = next(ans for ans in vlm_answers if ans in most_frequent_answers)
    result_sentence = f"For these videos, the tool's sequential answers are: {{{', '.join(vlm_answers)}}}. The inferred answer should be {most_frequent_answer}."
    return result_sentence