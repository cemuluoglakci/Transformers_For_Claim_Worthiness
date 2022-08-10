import sys
if ".." not in sys.path:
    sys.path.append('../')
import streamlit as st
import openai
import pandas as pd
import numpy as np
import os
import torch
import wandb

#Custom modules
import utils
from utils import custom_models, early_stopping, worthiness_checker, constants

st.set_page_config(
    page_icon=':shark:',
    page_title='Check-Worthiness',
    initial_sidebar_state='auto',
    layout="wide",
)

st.markdown(""" <style>
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.title('Let me check your Check-Worthiness')

openai.api_key = st.secrets["api_key"]
fine_tuned_model = st.secrets["fine_tuned_model"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parent_dir = os.path.dirname(os.path.abspath(os.getcwd()))
seed_list = [7, 42] # seed_list = [7, 42, 127]
fold_count = 3 #5
patience=5

constants = constants.Constants()
constants.device = device
constants.parent_dir = parent_dir
constants.seed_list = seed_list
constants.fold_count = fold_count
constants.patience = patience

api = wandb.Api()
best_sweep = '2afv0m0i'
sweep = api.sweep("cemulu/Transformers_For_ClaimWorthiness/" + best_sweep)
best_run = sweep.best_run()
best_run.summary.get("avg_val_mAP")
worthiness_checker = worthiness_checker.WorthinessChecker(best_run, constants)

model_file_name = 'vinai_bertweet-covid19-base-uncased_0.7651173954688729.pt'
PATH = os.path.join(parent_dir, 'Model', model_file_name)
worthiness_checker.load_model(PATH)

def check_worthiness(tweet):
    tweet = tweet + "\n\n###\n\n"
    result = openai.Completion.create(model = fine_tuned_model, prompt=str(tweet), max_tokens=10, temperature=0)['choices'][0]['text'] 
    print('- ', tweet, ': ', result)
    return result


input = st.text_input("",placeholder='Is this statement Chech-Worthy?')
if st.button('Submit'):
    st.write('**Output**')
    st.write(f"""---""")
    with st.spinner(text='In progress'):
        report_text = check_worthiness(input)
        st.markdown('GPT3 says: '+report_text)
    with st.spinner(text='In progress'):
        probability = worthiness_checker.predict(input)
        st.markdown('BERTweet says: ' + str(probability))
c30, c31, c32, c33, c34 = st.columns([1,1,2,1,1])
with c32:
    st.image("./media/pandaai.png", use_column_width=True)