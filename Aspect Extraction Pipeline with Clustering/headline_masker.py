# !pip install gliner

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gliner import GLiNER
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_masks = {"Person" : "PERSON", 
               "Organization":"ORGANIZATION", 
               "Location" : "LOCATION", 
               "Date": "DATE", 
               "Event":"EVENT", 
               "Monetary Value":"MONEY", 
               "Product":"PRODUCT",
               "Ordinal Number": "ORDINAL", 
               "Percentage": "PERCENTAGE"} 
labels = list(label_masks.keys())

def mask_entities_gliner(model, text):
    entities = model.predict_entities(text, labels,label_masks, threshold=0.5)
    masked_text = text
    # Replace each entity with its corresponding label in square brackets
    for entity in entities:
        masked_text = masked_text.replace(entity['text'], f'{label_masks[entity["label"]]}')
    return masked_text

# %% [code]


def masker(headlines_df = None, gliner_model = "urchade/gliner_medium-v2.1"):
    model = GLiNER.from_pretrained(gliner_model).to(device)

    masked_headlines = []
    
    # Get the total number of rows
    total_rows = len(headlines_df)
    
    # Iterate through the DataFrame row by row and process it
    for idx, row in tqdm(headlines_df.iterrows(),desc = "Masking Headlines",total=total_rows):
        headline = row['headline']
        masked_headline= mask_entities_gliner(model, headline)
        
        # Append the masked headline to the list
        masked_headlines.append(masked_headline)

    return masked_headlines
    

if __name__ == "__main__":
    
    import pandas as pd
    root= "/kaggle/input/nlp-nifty50/Nifty50_news_data(2020Jan_2024April) (1).csv"
    df= pd.read_csv(root)
    
    masker(df)
