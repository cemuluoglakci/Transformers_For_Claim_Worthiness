import streamlit as st
import openai
import pandas as pd
import numpy as np

st.title('Check your Check-Worthiness')

openai.api_key = st.secrets["api_key"]
fine_tuned_model = "ada:ft-personal-2022-08-09-12-35-53"

"You can get the Id but not the Key"
st.write("Model Id:", st.secrets["finetune_id"])

def check_worthiness(tweet):
    tweet = tweet + "\n\n###\n\n"
    result = openai.Completion.create(model = fine_tuned_model, prompt=str(tweet), max_tokens=10, temperature=0)['choices'][0]['text'] 
    print('- ', tweet, ': ', result)
    return result


input = st.text_input('Input:')
if st.button('Submit'):
    st.write('**Output**')
    st.write(f"""---""")
    with st.spinner(text='In progress'):
        report_text = check_worthiness(input)
        st.markdown(report_text)