import ollama
import os
import streamlit as st
import json
import csv
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr

model = "llama3.2:3b"
input_file = "./data/grocery.txt"
output_txt = "./data/categorized_grocery.txt"
custom_categories_file = "./data/custom_categories.json"

language = input("Enter language code (default: en for English): ").strip().lower() or "en"

def get_grocery_items():
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        print("File not found! You can enter items manually or use voice input.")

        choice = input("Do you want to enter groceries via voice? (y/n): ").strip().lower()
        if choice == 'y':
            return get_voice_input()
        else:
            return input("Enter grocery items (comma-separated): ").strip()

def get_voice_input():
    fs = 44100  # Sample rate (44.1 kHz)
    duration = 5  # Seconds to record
    print("Speak your grocery items...")
    
    # Record the audio
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    # Save the recorded audio as a .wav file
    wav.write("input_audio.wav", fs, myrecording)
    
    # Use speech recognition to transcribe the audio
    recognizer = sr.Recognizer()
    with sr.AudioFile("input_audio.wav") as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized items: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the speech.")
        except sr.RequestError:
            print("Error connecting to speech recognition service.")
    return ""

def load_custom_categories():
    if os.path.exists(custom_categories_file):
        with open(custom_categories_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

items = get_grocery_items()

custom_categories = load_custom_categories()

prompt = f"""
You are an assistant that categorizes and sorts grocery items.

Here is a list of groceries:
{items}

Please:
1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Use the following custom categories if applicable: {json.dumps(custom_categories, indent=2)}
4. Present the categorized list in a clear and organized manner, using bullet points or numbering.
5. Respond in {language}.
"""

try:
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    generated_text = response.get("message", {}).get("content", "")

    if not generated_text.strip():
        raise ValueError("The model returned an empty response.")

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(generated_text.strip())

    print(f"✅ Categorized grocery list saved in: \n- {output_txt}\n")

except Exception as e:
    print(f"❌ An error occurred: {str(e)}")
