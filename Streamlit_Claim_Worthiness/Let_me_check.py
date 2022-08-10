import streamlit as st
import openai
import pandas as pd
import numpy as np

st.set_page_config(
    page_icon=':shark:',
    page_title='Check-Worthiness',
    initial_sidebar_state='auto',
    layout="wide",
)
BACKGROUND_COLOR = 'white'
COLOR = 'black'
def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .appview-container .sidebar-content {{
                    padding-top: {padding_top}rem;
                }}
                .appview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .appview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

st.markdown(""" <style>
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.markdown("""<style>
        .appview-container .main .block-container{{
        max-width: {percentage_width_main}%;
        padding-top: {1}rem;
        padding-right: {1}rem;
        padding-left: {1}rem;
        padding-bottom: {1}rem;
    }}

        .uploadedFile {{display: none}}
        footer {{visibility: hidden;}}
</style>""", unsafe_allow_html=True)

st.markdown(""" <style>
.appview-container .main .block-container{{
        padding-top: {padding_top}rem;    }}
</style> """, unsafe_allow_html=True)




st.title('Let me check your Check-Worthiness')

openai.api_key = st.secrets["api_key"]
fine_tuned_model = st.secrets["fine_tuned_model"]

c30, c31, c32 = st.columns([2,1,2])
with c31:
    st.image("./media/pandaai.png", use_column_width=True)

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