# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-mpnet-base-v2").to(device)

def generate_embeddings(masked_headlines):
    embeddings = []
    
    # Iterate through the input column and generate embeddings with progress bar
    for masked_headline in tqdm(masked_headlines, total=len(masked_headlines)):
        embedding = model.encode([masked_headline],convert_to_tensor=True)
        embeddings.append(embedding.cpu().numpy())
    
    return embeddings