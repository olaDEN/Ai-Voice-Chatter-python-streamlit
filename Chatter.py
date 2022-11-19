from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import transformers
from gtts import gTTS
import speech_recognition as sr  
from io import BytesIO
import pygame
import time 
# # Source: https://github.com/olaDEN

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
 model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
 return tokenizer, model

tokenizer, model = load_data()

st.title("Ai-ChatterBot")

st.write("Hey there \U0001F44B \U0001F603 please click the \"Speak\" button and chat with me \U0001F60A ")

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
mic = sr.Microphone()

speak = st.button("Speak")

with mic as source:
    if speak:  
        audio = recognizer.adjust_for_ambient_noise(source, duration=0.2)
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source, timeout=3, phrase_time_limit=8)
        try:
            # take user input
            text = recognizer.recognize_google(audio)
            text = text.lower()

            # refreshing every after 5 times for better performance
            if 'count' not in st.session_state or st.session_state.count == 6:
                st.session_state.count = 0 
                st.session_state.chat_history_ids = None
                st.session_state.old_response = ''
            else:
                st.session_state.count += 1

            # encode the input and add end of string token
            input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")

            #concatenate new user input with chat history (if there is)
            bot_input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1) if st.session_state.count > 1 else input_ids

            # generate a bot response
            st.session_state.chat_history_ids = model.generate(
            bot_input_ids, max_length=2000,
            pad_token_id=tokenizer.eos_token_id,  
            no_repeat_ngram_size=3,       
            do_sample=True, 
            top_k=100, 
            top_p=0.7,
            temperature=0.8
            )
                
            #print the output
            output = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            #enhancing performance
            if st.session_state.old_response == output:
                bot_input_ids = input_ids
 
                st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=2000, pad_token_id=tokenizer.eos_token_id)
                output = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            @st.cache
            def speak(text, language='en'):
                mp3_fo = BytesIO()
                tts = gTTS(text, lang=language)
                tts.write_to_fp(mp3_fo)
                pygame.mixer.music.load(mp3_fo, 'mp3')
                time.sleep(0.1)
                pygame.mixer.music.play()
                # return mp3_fo
            pygame.init()
            pygame.mixer.init()
            # sound.seek(0)
            st.markdown(f"## ChatterBot reply:")
            output2 = speak(output)
            st.success(f"{output}") 
            st.warning(f"You said: {text}")
            st.session_state.old_response = output
        except sr.UnknownValueError:
            st.write("Speak again please")
        except sr.RequestError:
            st.write("Speech Service down")
          
# # Source: https://github.com/olaDEN
