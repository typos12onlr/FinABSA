# %% [code]
# %% [code]
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-12-29T15:42:51.809753Z","iopub.execute_input":"2024-12-29T15:42:51.810155Z","iopub.status.idle":"2024-12-29T15:42:52.256551Z","shell.execute_reply.started":"2024-12-29T15:42:51.810127Z","shell.execute_reply":"2024-12-29T15:42:52.255349Z"}}
import pandas as pd
import json
from pprint import pprint
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-12-29T15:42:54.362896Z","iopub.execute_input":"2024-12-29T15:42:54.363214Z","iopub.status.idle":"2024-12-29T15:42:54.368915Z","shell.execute_reply.started":"2024-12-29T15:42:54.363190Z","shell.execute_reply":"2024-12-29T15:42:54.367828Z"}}
def sample_from_clusters(df, cluster_column, sample_percentage=0.1):
    sampled_data = []

    for cluster_label, group in df.groupby(cluster_column):
        cluster_size = len(group)
        sample_size = max(5, int(cluster_size * sample_percentage))  # Ensure at least 1 sample
        
        sampled_group = group.sample(sample_size, random_state=1)  
        sampled_data.append(sampled_group)

    return pd.concat(sampled_data, ignore_index=True)

import json
import os

def create_update_prompts_json(mode="create", system_prompt="", user_prompt_template="", filename="prompts.json"):
    if mode not in ["create", "update"]:
        raise ValueError("Mode must be either 'create' or 'update'")

    if mode == "update" and not os.path.exists(filename):
        raise ValueError(f"Cannot update: File {filename} does not exist")
    
    if mode == "create":
        # Create new dictionary with provided values
        prompts = {
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template
        }
    else:  # mode == "update"
        # Read existing file
        try:
            with open(filename, "r") as json_file:
                prompts = json.load(json_file)
                
            # Update only non-empty values
            if system_prompt:
                prompts["system_prompt"] = system_prompt
            if user_prompt_template:
                prompts["user_prompt_template"] = user_prompt_template
                
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {filename}")
    
    # Write to file
    with open(filename, "w") as json_file:
        json.dump(prompts, json_file, indent=4)
    
    print(f"Successfully {'created' if mode == 'create' else 'updated'} JSON file.")

def read_prompts(filename):
    try:
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        print("Data read from JSON file")
    except json.JSONDecodeError:
        print("The JSON file is empty or invalid.")
        data = {}  # Default to an empty dictionary
    return data
    
from typing import List
from pydantic import BaseModel

class Aspect(BaseModel):
    snippet: str
    level_1_aspect: str
    level_2_aspect: str  
    reasoning: str

class AspectExtraction(BaseModel):
    sentence: str
    info: List[Aspect]


def invoke_llm(client = None, prompts = {}, sentence: str = "", aspects: List[str] = [""], model = "llama3.1:70b") -> AspectExtraction:
    system_prompt = prompts["system_prompt"]
    user_prompt_template = prompts["user_prompt_template"]
    user_prompt = user_prompt_template.format(sentence=sentence, aspects=aspects)

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=AspectExtraction,
        temperature=0,  # Maximum determinism
        seed=42,  # Fixed seed for reproducibility
        max_tokens=600  # Adequate length for detailed responses
    )

    return completion.choices[0].message.parsed


def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def extract_cluster_aspects(client = None, descriptions = None, prompts_file = "prompts.json", model = "llama3.1:70b"):
    prompts = read_prompts(prompts_file)
    level_1_aspects = []
    level_2_aspects = []
    aspects = []
    raw_responses = []  # Store all raw responses
    
    for idx, description in enumerate(descriptions):
        # Get response
        response = invoke_llm(client = client, prompts = prompts, sentence = description, aspects= aspects, model = model)
        data = response.model_dump()
        pprint(data)
        
        # Save individual response
        save_data(data, f'raw_response_cluster_{idx}.json')
        raw_responses.append(data)  # Keep in memory too
        
        # Process aspects
        for entry in data["info"]:
            if entry["level_1_aspect"] not in level_1_aspects:
                level_1_aspects.append(entry["level_1_aspect"])
            if entry["level_2_aspect"] not in level_2_aspects:
                level_2_aspects.append(entry["level_2_aspect"])
            aspect = "/".join([entry["level_1_aspect"], entry["level_2_aspect"]])
            if aspect not in aspects:
                aspects.append(aspect)
        
        # Save current state
        current_state = {
            "level_1_aspects": level_1_aspects,
            "level_2_aspects": level_2_aspects,
            "aspects": aspects,
            "raw_responses": raw_responses
        }
        save_data(current_state, 'cluster_extraction_progress.json')
    
    return level_1_aspects, level_2_aspects, aspects