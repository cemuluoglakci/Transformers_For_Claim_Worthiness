import sys
import os
if ".." not in sys.path:
    sys.path.append('../')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import streamlit as st
import openai
import wandb

#Custom modules
from utils.constants import Constants
from utils.worthiness_checker import WorthinessChecker

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
parent_dir = os.path.dirname(os.path.abspath(os.getcwd()))

constants = Constants()
constants.parent_dir = parent_dir

api = wandb.Api()

def check_worthiness(tweet):
    print('arrived gpt')
    tweet = tweet + "\n\n###\n\n"
    result = openai.Completion.create(model = fine_tuned_model, prompt=str(tweet), max_tokens=10, temperature=0)['choices'][0]['text'] 
    print('- ', tweet, ': ', result)
    return result

@st.experimental_singleton
def get_bert_model():
    best_sweep = '2afv0m0i'
    sweep = api.sweep("cemulu/Transformers_For_ClaimWorthiness/" + best_sweep)
    best_run = sweep.best_run()
    best_run.summary.get("avg_val_mAP")
    checker = WorthinessChecker(best_run, constants)

    model_file_name = 'vinai_bertweet-covid19-base-uncased_0.7651173954688729.pt'
    PATH = os.path.join(parent_dir, 'Model', model_file_name)
    checker.load_model(PATH)
    return checker

input = st.text_input("Is this statement Chech-Worthy?",placeholder='Is this statement Chech-Worthy?')
button = st.button('Submit')

c30, c31, c32, c33, c34 = st.columns([1,1,2,1,1])
with c32:
    st.image("./media/pandaai.png", use_column_width=True)

if button:
    with st.spinner(text='In progress'):
        report_text = check_worthiness(input)
        st.markdown('**GPT3 says:** '+report_text)
    with st.spinner(text='In progress'):
        bertweet_checker = get_bert_model()
        probability = bertweet_checker.prediction_expression(input)
        st.markdown('**BERTweet says:** ' + str(probability))
