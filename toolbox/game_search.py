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



### GAME SEARCH

import re

def extract_match_info(input_text):
    INSTRUCTION = """
    You are a helpful assistant that extracts structured information from natural language text about football matches. I will give you a sentence about a football match, and you need to extract the following information: league, season, date, time, and two teams. The output must strictly follow the format below:

    league: (england_epl, germany_bundesliga, europe_uefa-champions-league, italy_serie-a, france_league-1, spain_laliga, or unknown)
    season: xxxx-xxxx
    date: xxxx-xx-xx
    year: xxxx
    month: xx
    day: xx
    time: xx:xx (which means when this game kick-off, not the game timestamp of certain event)
    score: x - x (if score is not determined, write 'unknown' for only in this attribute)
    team1: yyy
    team2: yyy

    All above 'x' means a digit!! 'yyy' means a string.

    To be noted, if you can determine only one team, please assign the team to team1 and leave team2 as 'unknown'. If any information is missing or uncertain, write 'unknown'. You have to use the exactly same name of teams as provided in the input text. Do not output any other words.
    For other attributes, if any information is missing or uncertain, write 'unknown'. As for date, you should record in the form of xxxx-xx-xx if you can get the clear date; Meanwhile, as for year, month, day, you need capture as more information point to this game as possible, including year, month, and day, and record them in numbers.
    Do not guess any information. For example if year is not said clearly, don't guess the year through season. Only use the information provided in the input text. Do not output any other words.
    """
    default_dict = {
        "league": "unknown",
        "season": "unknown",
        "date": "unknown",
        "year": "unknown",
        "month": "unknown",
        "day": "unknown",
        "time": "unknown",
        "score": "unknown",
        "team1": "unknown",
        "team2": "unknown",
    }
    llm_output = workflow(input_text, INSTRUCTION)
    pattern = re.compile(
        r"^(\w+):\s*(.*?)\s*$",
        re.MULTILINE 
    )
    
    match = pattern.findall(llm_output)
    info = {key: value for key, value in match}
    return info if info else default_dict

import pandas as pd

def retrieve_candidates(info, csv_path=os.path.join(PROJECT_PATH, "database/Game_dataset_csv/game_database.csv")):

    df = pd.read_csv(csv_path)
    conditions = []
    if info["league"] != "unknown":
        conditions.append(df["league"] == info["league"])
    if info["season"] != "unknown":
        conditions.append(df["season"] == info["season"])
    if info["year"] != "unknown":
        conditions.append(df["year"] == int(info["year"]))
    if info["month"] != "unknown":
        conditions.append(df["month"] == int(info["month"].lstrip('0')))
    if info["day"] != "unknown":
        conditions.append(df["day"] == int(info["day"].lstrip('0')))
    if info["time"] != "unknown":
        conditions.append(df["time"] == info["time"])
 
    if conditions:
        initial_filtered_df = df[pd.concat(conditions, axis=1).all(axis=1)]
    else:
        initial_filtered_df = df
    team_fields = ["team1", "team2"]
    team_values = [info[field] for field in team_fields]
    
    if any(value is not None for value in team_values):
        team_conditions = []
        if info["team1"] and info["team2"]:
            team_conditions.append(
                (df["home_team"].str.replace(" ", "").str.contains(info["team1"].replace(" ", ""), case=False, na=False) &
                 df["away_team"].str.replace(" ", "").str.contains(info["team2"].replace(" ", ""), case=False, na=False)) |
                (df["home_team"].str.replace(" ", "").str.contains(info["team2"].replace(" ", ""), case=False, na=False) &
                 df["away_team"].str.replace(" ", "").str.contains(info["team1"].replace(" ", ""), case=False, na=False))
            )
        elif info["team1"]:
            team_conditions.append(
                df["home_team"].str.replace(" ", "").str.contains(info["team1"].replace(" ", ""), case=False, na=False) |
                df["away_team"].str.replace(" ", "").str.contains(info["team1"].replace(" ", ""), case=False, na=False)
            )
        elif info["team2"]:
            team_conditions.append(
                df["home_team"].str.replace(" ", "").str.contains(info["team2"].replace(" ", ""), case=False, na=False) |
                df["away_team"].str.replace(" ", "").str.contains(info["team2"].replace(" ", ""), case=False, na=False)
            )
        
        if team_conditions:
            final_filtered_df = initial_filtered_df[pd.concat(team_conditions, axis=1).any(axis=1)]
        else:
            final_filtered_df = initial_filtered_df
    else:
        final_filtered_df = initial_filtered_df
    
    if len(final_filtered_df) > 10:
        final_filtered_df = None
    
    return initial_filtered_df, final_filtered_df

def finalize_candidate_selection(candidates, candidates_with_team, info, question):

    if candidates is None or len(candidates) == 0:
        return "We did not find the match you mentioned in the database."
    
    if len(candidates) == 1:
        file_path = candidates.iloc[0]["file_path"]
        return f"The game information file path is: {file_path}"
    if len(candidates_with_team) == 1:
        file_path = candidates_with_team.iloc[0]["file_path"]
        return f"The game information file path is: {file_path}"
    
    if len(candidates) > 1:
        prompt = f"""
        You are a helpful assistant that selects the most likely match from a list of candidates based on the given information. Now we need to retrieve a file path for the most probable match from the database from the question: "{question}".
        
        Such question has been transformed to the original query information as:
        
        {info}
        
        Here are the candidate matches:
        """
        
        for i, row in candidates.iterrows():
            prompt += f"""
            Candidate {i + 1}:
            - League: {row['league']}
            - Season: {row['season']}
            - Date: {row['date']}
            - Year: {row['year']}
            - Month: {row['month']}
            - Day: {row['day']}
            - Time: {row['time']}
            - Score: {row['score']}
            - Home Team: {row['home_team']}
            - Away Team: {row['away_team']}
            - file_path: {row['file_path']}
            """
        
        prompt += """
        Based on the original query information and the candidate matches above, is there a match that is significantly more likely than the others? 

        Firstly, you should exclude those candidates in the following situation:
        1. If **any of the team's name in original query information** is sure not to be in team names from candidates, such candidate cannot be returned anymore, you cannot let such candidate take place in your return answer. 
        2. For example, if the original query information contains "Chelsea" and "West Ham", but candidates contains "chelsea FC" and "Liverpool", since such candidate cannot be returned anymore since West Ham is not in candidate information.
        3. For example, if the original query information contains "Chelsea" and "West Ham", but candidates contains "Chelsea FC" and "West Ham United", since such candidate is still possible to be returned since both team names are in candidate information.
        4. For example, if the original query information contains only "Chelsea", but candidates contains "Bayern Munich" and "Real Madrid", since such candidate cannot be returned since Chelsea is not in candidate information.
        
        After considering the above situation and exclude those candidate having team name unmatched, you should consider the following two situations:

        1. If there are still **obviously** probable answer with all known information correct, please return the file path of that match EXACTLY in the following format:
        "The given information seems incomplete, but we found the most probable match in the database with this file path: [The file path of the **hugely most probable** match]. [Here give some recommendation to complete the information if possible, for example, provide the date or the score of the match, or which team is the home/away team .etc. Use simple and clear words here.]" 

        2. If no match is significantly more likely among all the candidates, please return all candidate matches with information of league, season, date, time, score, home_team, away_team, venue and referee (without file path), and explain that the information provided is too vague. For this situation you only need to summarize with a little bit the games and give a brief reply with some short sentences.
        """
        
        llm_output = workflow(prompt, "You are an soccer expert that selects the most likely match from a list of candidates based on the given information.")
        
        return llm_output

def GAME_SEARCH(query, materials=None):
    info = extract_match_info(query)
    candidates, candidates_without_team = retrieve_candidates(info)
    result = finalize_candidate_selection(candidates, candidates_without_team, info, query)
    return result