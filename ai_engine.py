import pickle
import numpy as np
import tensorflow as tf
import torch
from textblob import TextBlob
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

print("Loading Production Assets...")
model = tf.keras.models.load_model('models/support_ai_model.keras')

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('models/response_mapping.pkl', 'rb') as f:
    response_mapping = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Global variable to store the conversational memory
chat_history_ids = None

def analyze_sentiment(text):
    analysis = TextBlob(text).sentiment.polarity
    if analysis > 0.2: return "Positive"
    if analysis < -0.2: return "Negative"
    return "Neutral"

def get_ai_response(user_text):
    global chat_history_ids # Access the memory
    
    clean_text = user_text.lower().strip()

    # ==========================================
    # LAYER 1: The Rule-Based Interceptor
    # ==========================================
    if clean_text in ["who are you", "what are you", "are you a robot", "are you a human"]:
        return {
            "intent": "bot_identity",
            "confidence": 1.0,
            "response": "I am an AI Support Assistant! I specialize in tracking orders and handling refunds, but I love chatting too.",
            "sentiment": "Neutral"
        }

    # ==========================================
    # LAYER 2: TensorFlow Support Logic
    # ==========================================
    prediction = model.predict(tf.constant([user_text], dtype=tf.string), verbose=0)
    predicted_index = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_index])
    
    intent = label_encoder.inverse_transform([predicted_index])[0]
    sentiment = analyze_sentiment(user_text)
    
    # We raise the threshold to 0.85. The AI must be VERY sure it's a business question.
    if confidence > 0.85:
        response = response_mapping.get(intent, "I can assist with that.")
        
        # Empathy logic
        if sentiment == "Negative":
            response = "I completely understand why that's frustrating. Let's get this sorted. " + response
            
        # Reset the small-talk memory since we are talking business now
        chat_history_ids = None 
            
        return {
            "intent": intent,
            "confidence": confidence,
            "response": response,
            "sentiment": sentiment
        }
        

    # ==========================================
    # LAYER 3: LLM Conversational Fallback with Memory
    # ==========================================
    else:
        # Translate the new user input
        new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
        
        # Combine the new input with the past conversation history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Explicitly create an attention mask to satisfy Hugging Face
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

        # Generate a response with Anti-Looping safeguards (Sampling)
        chat_history_ids = chat_model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,     # Stops it from getting stuck in robotic loops
            top_k=50,           # Restricts choices to the top 50 logical words
            top_p=0.95,         # Nucleus sampling for natural phrasing
            temperature=0.75    # Controls the "creativity" (0.0 is rigid, 1.0 is chaotic)
        )
        
        # Extract ONLY the bot's newly generated text
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Failsafe if the LLM glitches out
        if not response.strip():
            response = "That's interesting! By the way, let me know if you need help with your orders or account."
            chat_history_ids = None # Reset memory on glitch

        return {
            "intent": "general_conversation",
            "confidence": confidence,
            "response": response,
            "sentiment": sentiment
        }