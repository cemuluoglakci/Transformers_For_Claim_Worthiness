import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import openai

#Custom modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.constants import Constants
from utils.worthiness_checker import Predictor

openai.api_key = st.secrets["api_key"]

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

def check_worthiness(tweet):
    tweet = tweet + "\n\n###\n\n"
    result = openai.Completion.create(model = st.secrets["fine_tuned_model"],
     prompt=str(tweet), max_tokens=10, temperature=0,logprobs=5)['choices'][0]
    answer = result['text']
    probability = pd.DataFrame([result["logprobs"]["top_logprobs"][0]]).T.apply(lambda x: 100*np.e**x).max().item() 
    return '{} and it is {:.4f}% sure '.format(answer, probability)

@st.experimental_singleton
def get_bert_model(model_name):
    constants = Constants()
    constants.parent_dir = os.path.dirname(os.path.abspath(os.getcwd()))
    best_run = st.secrets[model_name+"_best_model"]
    checker = Predictor(best_run, constants)
    checker.load_model_online(st.secrets[model_name+"_url"])
    return checker

st.title('Let me check your Check-Worthiness')

input = st.text_input("Is this statement Chech-Worthy?",placeholder='Is this statement Chech-Worthy?')
button = st.button('Submit')

c30, c31, c32, c33, c34 = st.columns([1,1,2,1,1])
with c32:
    st.image("./Streamlit_Claim_Worthiness/media/pandaai.png", use_column_width=True)

if button:
    with st.spinner(text='In progress'):
        report_text = check_worthiness(input)
        st.markdown('**GPT3 says:** '+report_text)
    with st.spinner(text='In progress'):
        bertweet_checker = get_bert_model("bertweet")
        prediction_string = bertweet_checker.prediction_expression(input)
        st.markdown('****BERTweet says:**** ' + prediction_string)
    with st.spinner(text='In progress'):
        bertweet_checker = get_bert_model("roberta")
        prediction_string = bertweet_checker.prediction_expression(input)
        st.markdown('****RoBERTa says:**** ' + prediction_string)
    with st.spinner(text='In progress'):
        bertweet_checker = get_bert_model("bert")
        prediction_string = bertweet_checker.prediction_expression(input)
        st.markdown('****BERT says:**** ' + prediction_string)

