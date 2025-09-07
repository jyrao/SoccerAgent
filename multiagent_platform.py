import csv
import re, os
import ast
from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import sys
sys.path.append('YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE/pipeline')



######################################
##                                  ##
##   PHASE 1 - Task Decomposition   ##
##                                  ##
######################################


######################## Parameters ########################
PROJECT_PATH = "YOUR_FOLDER_PATH_TO_SOCCERAGENT_CODEBASE"
from toolbox import *

toolbox_functions = {
    "Textual Entity Search": TEXTUAL_ENTITY_SEARCH,
    "Textual Retrieval Augment": TEXTUAL_RETRIEVAL_AUGMENT,
    "Game Search": GAME_SEARCH,
    "Game Info Retrieval": GAME_INFO_RETRIEVAL,
    "Match History Retrieval": MATCH_HISTORY_RETRIEVAL,
    "Entity Recognition": FACE_RECOGNITION,
    "Number Recognition": JERSEY_NUMBER_RECOGNITION,
    "Camera Detection": CAMERA_DETECTION,
    "Segment": SEGMENT,
    "Shot Change": SHOT_CHANGE,
    "Action Classifier": ACTION_CLASSIFICATION,
    "Commentary Generation": COMMENTARY_GENERATION,
    "Jersey Color Relevent VQA": JERSEY_COLOR_VLM,
    "Vision Language Model": VLM,
    "Replay Grounding": REPLAY_GROUNDING,
    "Score and Time Recognition": SCORE_TIME_DETECTION,
    "Frame Selection": FRAME_SELECTION,
    "Foul Recognition": FOUL_RECOGNITION
}

import os
os.system('')


######################## Helper Functions ########################

def load_toolbox(file_path):
    descriptions = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        # reader = csv.DictReader(csvfile)
        reader = csv.DictReader(csvfile, quotechar='"', skipinitialspace=True)
        i = 0
        for row in reader:
            i += 1
            tool_name = row['name']
            ability = row['ability']
            query_input = row['query input']
            material_input = row['material input']
            output = row['output']
            remark = row['remark']
            
            description = f"=== Tool Description for TOOL{i} ===\n"
            # description += f"Name: TOOL{i}\n"
            description += f"Name: {tool_name}\n"
            description += f"Ability: {ability}\n"
            description += f"Query Input: {query_input}\n"
            description += f"material Input: {material_input}\n"
            description += f"Output: {output}\n"
            description += f"Remark: {remark}\n"
            
            descriptions.append(description)
    
    return descriptions

def load_toolbox_str(file_path=os.path.join(PROJECT_PATH, "pipeline/toolbox.csv")):
    toolbox = ""
    for i in load_toolbox(file_path):
        toolbox += f"{i}\n"
    # print(toolbox)
    return toolbox


def csv_to_task_string(csv_path=os.path.join(PROJECT_PATH, "pipeline/tasks.csv")):
    result = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file, quotechar='"', skipinitialspace=True)
        next(reader) 
        for i, row in enumerate(reader, start=1):
            if not row:  
                continue
                
            task_str = f"Task{i}: ​**{row[0]}** {row[1]}"
            
            chain = None
            for j in range(2, 5):
                if j < len(row) and row[j].strip():
                    chain = row[j]
                    break
            
            if chain:
                task_str += f"\nRecommended chain: {chain}"
            
            result.append(task_str + "\n")
    
    return "\n".join(result)

def generate_prompt(taskdecompositionprompt, query, additional_material):
    prompt = taskdecompositionprompt
    prompt += f"\nQuery: {query}\n"
    prompt += f"Adittional Material: {additional_material}\n"
    return prompt

def workflow(input_text, Instruction, follow_up_prompt=None, api_key="your-deepseek-api-key", max_tokens_followup=1500):

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
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
    
def parse_input(input_str):
    known_info = re.findall(r'\$(.*?)\$', input_str)
    
    tool_chain = re.findall(r'\*(.*?)\*', input_str)
    
    return (known_info, tool_chain)

def generate_prompt_execution(query, material, response, toolbox):
    prompt_execution = f"""As a multi-agent core in the Soccer Question Answering Assistant, you are required to execute the following tool chain to answer the question:

"{query}"

with the following additional material:

{material}

with the known info as:

{parse_input(response)[0]}

and you should execute the following tool chain to solve the question:

{parse_input(response)[1]}

As for the usage of the tools, you should follow the following references:

{toolbox}

For every tool above, we would input queries and materials into the tool for execution, the queries are in **text** form and the materials are in list with **file paths**. If no file path is suitable, you just write in 'None' You should determine the contents of materials and queries based on the context of the question, known info and tool descriptions.

For every steps of excution, you should return me with a clear statement of the goal of this step in the context of the overall analysis, the specific tool you are using, and the input variables you are using.

<Call>
    <Purpose>Brief, clear statement of this step’s goal in context of overall analysis</Purpose> 
    <Query>[Query/question here(string). IMPORTANT!!: Such query is highly relevant to the toolbox descriptions. you need to think carefully about your purpose this step and generate appropriate query.]</Query> 
    <Material>[Material list here(a string showing list form). Here as well, you need to think carefully considering the purpose and toolbox.]</Material>
    <Tool>[Tool name here(string)]</Tool>
</Call>

If it is the last step of the execution, you should return me with the following format:

<EndCall>
    <Purpose>Brief, clear statement of this step’s goal in context of overall analysis</Purpose> 
    <Query>[Query/question here(string)]</Query> 
    <Material>[Material list with file paths here(a string showing list form)]</Material>
    <Tool>[Tool name here(string)]</Tool>
</EndCall>

Every time you return me with the instruction as above, I will execute it and return you with the feedback of the execution in this format:

<StepResult>
    <Answer>[The results of this time's execution here(string)]</Answer>
</StepResult>

For every time of generation, you should follow the following rules:
    1. You should be clear about the tool name (must be chosen from toolbox), file path and query/question in the instruction. This part is important for me to understand the context of the execution. You cannot change any of the information in the instruction.
    2. If I have given you the feedback of the execution, you should analyze what you should write in the next call based on the feedback considering the tool chain I gave you and the task descriptions and tool descriptions. You should not repeat the same instruction again.
    3. If my prompt leaves you to generate the first call, you should directly return me with the call in the form from <> to </>. You should not add any other information in the instruction.
    4. Otherwise, if in the prompt I have given you some <StepResult>, you should consider the total process of the execution and continue to return me exactly with the form from <> to </>. You should not add any other information in the instruction.

Once again, I repreat that the question is:

"{query}"

with the following additional material:

{material}

with the known info as:

{parse_input(response)[0]}

and you should execute the following tool chain to solve the question:

{parse_input(response)[1]}

The following is all our execution history, now you can start with your call of first step:
    """

    return prompt_execution

######################################
##                                  ##
##   PHASE 2 - Tool Execution       ##
##                                  ##
######################################


def parse_call_response(model_reply):
    """
    解析 model_reply 中的 <Call> 部分，提取 <Tool>, <Query>, <Material> 的内容。
    """
    tool = re.search(r'<Tool>(.*?)</Tool>', model_reply, re.DOTALL)
    query = re.search(r'<Query>(.*?)</Query>', model_reply, re.DOTALL)
    material = re.search(r'<Material>(.*?)</Material>', model_reply, re.DOTALL)

    if not tool or not query or not material:
        # raise ValueError("Invalid <Call> format in model_reply")
        print ("Invalid <Call> format in model_reply")

    return tool.group(1).strip(), query.group(1).strip(), material.group(1).strip()

def execute_tool_call(tool_name, query, material, toolbox_functions):
    """
    根据 tool_name 从 toolbox_functions 中找到对应的函数并执行。
    """
    if tool_name not in toolbox_functions:
        # raise ValueError(f"Tool '{tool_name}' not found in toolbox_functions")
        print(f"Tool '{tool_name}' not found in toolbox_functions")

    tool_function = toolbox_functions[tool_name]
    execution_retults = None
    try:
        execution_retults = tool_function(query, material)
    except Exception as e:
        execution_retults = f"Error occurred: {str(e)}"
    return f"<StepResult>\n    <Answer>{execution_retults}</Answer>\n</StepResult>"

def generate_LLM_prompt(query):
    """
    生成调用 LLM 做问题回答的 prompt。
    """
    prompt = f"""According to the total process of above conversation, now give me your answer to the question '{query}'. You should retrun me with the following form:
    
Answer: [Your answer here]
Reasoning and Explanation: [Your reasoning and explanation here]

Please make sure that your answer is consistent with the total process of the execution and the tool chain I gave you. You should not change the tool chain and the process of the execution. You should only give me the answer and the reasoning and explanation of the answer without any other words.
    """
    return prompt



def execute_tool_chain(input_text, toolbox_functions, Instruction="You are a helpful multi-agent assistant that can answer questions about soccer.", api_key="sk-qsB0dn28i0QhQw58099e30554bA94b3dBc96D0287bEaC563"):
    client = OpenAI(api_key=api_key, base_url="https://az.gptplus5.com/v1")
    # Initialize the conversation history with the system instruction and user input
    conversation_history = [
        {"role": "system", "content": Instruction},
        {"role": "user", "content": input_text}
    ]
    total_process = ""
    while True:
        # Generate a response from the model
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation_history
        )
        # Get the model's reply
        model_reply = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": model_reply})
        total_process += model_reply
        tool, query, material = parse_call_response(model_reply)
        if tool != "LLM":
            material = ast.literal_eval(material) if material is not None else []
            user_execution = execute_tool_call(tool, query, material, toolbox_functions)
            conversation_history.append({"role": "user", "content": user_execution})
            total_process += user_execution
        # Check if the reply contains <EndCall>
        if "<EndCall>" in model_reply:
            if tool == "LLM":
                # Generate a prompt for the LLM to answer the question
                llm_prompt = generate_LLM_prompt(query)
                conversation_history.append({"role": "assistant", "content": llm_prompt})
                completion = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=conversation_history
                )
                # Get the model's reply
                model_reply = completion.choices[0].message.content
                total_process += model_reply
            else:
                tool, query, material = parse_call_response(model_reply)
                material = ast.literal_eval(material) if material is not None else []
                user_execution = execute_tool_call(tool, query, material, toolbox_functions)
                total_process += user_execution
            
            return total_process

######################## Some Basic Prompt Information ########################

toolbox_descriptions = load_toolbox_str()
tasks = csv_to_task_string()

TaskDecompositionPrompt = f"""
# Soccer Question Answering Assistant

## Task overview
You are a multi-modal agent that can answer questions about soccer knowledge.
For each question, you will reveive:
- A question about soccer considering different aspects of soccer
- You might also receive one or more video clip or image as context

Your task involves three sequential parts:
1. Problem Decomposition (Part 1) 
    - Identify available information 
    - Break down the question into sequential steps 
2. Sequential Tool Application (Part 2) 
    - Execute one tool at a time 
    - Record each tool’s output 
    - Continue until sufficient information is gathered 
3. Solution Synthesis (Part 3) 
    - Integrate all results 
    - Generate final answer


## Available Tools

For all the QA, you need to decompose them and 
Here are the tools that you can use to answer the questions:
{toolbox_descriptions}

## Common QA Tasks

Here are some common QA tasks that you might meet in the questions, for each types of questions, we provide the recommended tool chain for you to answer the questions:

{tasks}

To be noted, at this stage you only need to treat this question as open-ended QA task, you can use the common QA tasks as reference to decompose the question and identify the required tools.

## Response Format for Part 1
For each query, you should respond ONLY with: 
    Known Info: [list any categories explicitly mentioned in the query and material] 
    Tool Chain: [list required tools connected by ->]

## Examples

Query 1: "How does the viewpoint of the camera shift in the video?"
Adittional Material: "video": ["/remote-home/jiayuanrao/MatchRAG/test_QA/lzf/Q8/QA_material/video_once/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_135.mkv"]

Your response:
Known Info: [$VideoClip$]
Tool Chain: [*Shot Change* -> *Crop Video* -> *Camera Detection* -> *LLM*]

Query 2: "What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley?"
Adittional Material: None

Your response:
Known Info: [$GameContext$]
Tool Chain: [*Game Search* -> *Game info Retrieval* -> *Match History Retrieval* -> *LLM*]

Query 3: "How many goals did the player who forced a corner score for Borussia Dortmund's senior team?"
Adittional Material: "video": ["/remote-home/jiayuanrao/dataset/matchtime/video_clips/SN-Caption-test-align/germany_bundesliga_2015-2016/2015-11-08 - 17-30 Dortmund 3 - 2 Schalke/1_42_03.mp4"]

Your response:
Known Info: [$VideoClip$, $GameContext$]
Tool Chain: [*Vision Language Model* -> *Entity Recognition* -> *Text Retrieval Augment* -> *LLM*]

## Important Rules
1. You should only use the tools provided in the toolbox to answer the questions and provide the exact tool names.
2. Use exact item category names with $$ to represent the information categories.
3. Use exact tool category names with ** as shown above to represent the tools.
4. Only respond with Part 1 analysis - Parts 2 & 3 will be addressed in subsequent interactions.
5. Connect tools using -> symbol
6. Try your best to decompose the question and identify the required tools, you can first reference the common QA tasks to get some ideas. If the template fits the question, you can directly use the recommended tool chain. If not, you can try to decompose the question and identify the required tools.
"""

def EXECUTE_TOOL_CHAIN(query, material):
    prompt = generate_prompt(TaskDecompositionPrompt, query, material)
    res = workflow(input_text=prompt, Instruction="You are an expert in soccer.")
    result = execute_tool_chain(generate_prompt_execution(query, material, res, toolbox_descriptions), toolbox_functions)
    return result
