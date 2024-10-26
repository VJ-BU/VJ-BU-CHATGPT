# import Steamlit to use in python environment to build the web inteface
import streamlit as st

# importt GPT model & Tokenizer from transformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# method to import transformer is to add transformers in requirements & run command pip install -r requirements.txt

#I have written command by creating folder .streamlit and creating file secrets.toml
#[api_keys] openai_api_key = "your_openai_api_key"
# fetch key using commands
# key = [api_keys][your_openai_api_key_here]

### however, i am observing this command disappears after running app???? but app keeps running
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

# Import pitorch for computations
# method to import pytoch is to add tortch in requirements & run command pip install -r requirements.txt
import torch 

# Mention the title of the app 
st.title("Welcome to VJ's Chat App (includes both Creative vs Predictable Responses)")

# Load pre-trained GPT-2 model from Hugging Face
# ref: https://huggingface.co/docs/transformers/en/model_doc/gpt2
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load  tokenizer from Hugging Face
# ref: https://huggingface.co/docs/transformers/en/model_doc/gpt2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# function to generate chatgpt responces
# temperature variable is set in a way to use for adjustments
#  ref: https://huggingface.co/docs/transformers/en/model_doc/gpt2
def generate_response(prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'], 
        max_length=max_tokens, 
        do_sample=True, 
        temperature=temperature  
        # temperature will be set for two different types of responses
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# skipping the special tokens as response is showing random signs & symboles

# function is created to start a new chat with every request 
# Observation: if new chat is not created then app loads slowly
def Start_new_chat():
    if 'new_message' in st.session_state:
        del st.session_state['new_message'] 
# session_state command used to activate session
# reference: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
# del is used to delete the previous chats to start a new chat with new inputs

# show empty container initially to start the chat
#Insert a single-element container
# reference: https://docs.streamlit.io/develop/api-reference/layout/st.empty
message_container = st.empty()

# Need to use form command to receive users input
# w3school: https://www.w3schools.com/python/python_user_input.asp
#  use st.form command to create form and get users input
# https://docs.streamlit.io/develop/api-reference/execution-flow/st.form
with st.form(key="chat_form", clear_on_submit=True):
    users_input = st.text_input("To Start chatting, type your message here!!!:", key="users_input")

    #set the number of tokens to be used
    # use slider to set the tokens
    # ref: https://docs.streamlit.io/develop/api-reference/widgets/st.slider
    max_tokens = st.slider("Select the number of tokens for your response", min_value=1, max_value=500, value=100)
    # range 0 to 500
    
# setting default values: 100
    user_input_submit = st.form_submit_button("Send")

# Create If loop to handle input from the user and generate responses
if user_input_submit and users_input:
    # using 'and' to make sure oth user input and submit is executed
    
    # As mentioned earlier reset the chat on new user input
    Start_new_chat()
    
    # set lower temperature to generate creative resopnse
    # use generate_response function
    # reference: https://www.rdocumentation.org/packages/simglm/versions/0.8.9/topics/generate_response
    cr_response = generate_response(users_input, max_tokens=max_tokens, temperature=1.5)
    
    # set lower temperature to generate predictable resopnse
      # use generate_response function
    # reference: https://www.rdocumentation.org/packages/simglm/versions/0.8.9/topics/generate_response
    pr_response = generate_response(users_input, max_tokens=max_tokens, temperature=0.1)
    
  
    # using ** to highlight users input 
    # using \n command to start repsonce next line
    # https://docs.streamlit.io/develop/api-reference/text/st.markdown
    # https://discuss.streamlit.io/t/applying-custom-css-to-manually-created-containers/33428
    message_container.markdown(f"**Your input message:** <span style='color:blue;'>{users_input}</span>\n\n"
    # take users input and show message initially

    f"**Creative Bot's Response =>** {cr_response}\n\n"
    # show creative bot's response

    f"**Predictable Bot's Response =>** {pr_response}", unsafe_allow_html=True)
    # show predictable bot's response
