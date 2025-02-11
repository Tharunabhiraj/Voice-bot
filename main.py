import os
import json
import asyncio
import requests
import pandas as pd
from collections import deque
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import google.generativeai as genai
import pyaudio
import dash
import datetime
from dash import dcc, html, Output, Input
import plotly.express as px

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DEEPGRAM_API_KEY or not GEMINI_API_KEY:
    raise ValueError("❌ Missing API keys. Set DEEPGRAM_API_KEY and GEMINI_API_KEY.")
genai.configure(api_key=GEMINI_API_KEY)
SESSION_HISTORY = deque(maxlen=5)
LOG_FILE = "bot_logs.csv"
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["user_query", "bot_response", "timestamp", "response_time"])
    df.to_csv(LOG_FILE, index=False)

def load_knowledge_base(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)["question"].tolist()
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return f.readlines()
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)["questions"]
    return []
KNOWLEDGE_BASE = load_knowledge_base(r"C:\Users\Admin\Downloads\Voice-bot\question,response.txt")

class NLPProcessor:
    """Extracts key queries from past user questions to reduce token consumption."""

    def __init__(self, knowledge_base):
        self.vectorizer = TfidfVectorizer()
        self.knowledge_base = knowledge_base
        self.vectors = self.vectorizer.fit_transform(knowledge_base)

    def get_top_relevant_queries(self, user_query, top_n=3):
        """Finds the most relevant questions from the knowledge base."""

        user_vector = self.vectorizer.transform([user_query])
        scores = (user_vector * self.vectors.T).toarray()[0]
        top_indices = scores.argsort()[-top_n:][::-1]
        return [self.knowledge_base[i] for i in top_indices if scores[i] > 0]

nlp_processor = NLPProcessor(KNOWLEDGE_BASE)

class GeminiChatbot:
    """Handles AI responses using Google Gemini API with memory optimization."""
    def log_interaction(self, user_query, bot_response, response_time):
        """Logs user query and AI response for dashboard."""
        data = {
            "user_query": user_query,
            "bot_response": bot_response,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "response_time": response_time
        }
        df = pd.DataFrame([data])
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    def generate_response(self, user_input):
        """Generates AI response using past session history and NLP filtering."""
        start_time = datetime.datetime.now()  # 🔹 Start time before calling AI

        relevant_queries = nlp_processor.get_top_relevant_queries(user_input,top=3)

        prompt = "You are a helpful hotel assistant. Answer based on past conversations and FAQs:\n"
        for i, past in enumerate(SESSION_HISTORY, 1):
            prompt += f"User {i}: {past['user']}\nAI {i}: {past['bot']}\n"
        
        prompt += f"\nUser Current Query: {user_input}\n"
        prompt += f"Related Questions from FAQ: {', '.join(relevant_queries)}\n"
        prompt += "AI Response:"

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        bot_response = response.text.strip() if response and response.text else "I am not sure about that."

        response_time = (datetime.datetime.now() - start_time).total_seconds()
        self.log_interaction(user_input, bot_response, response_time)
        return bot_response
    
class DeepgramSpeech:
    """Handles real-time speech recognition and text-to-speech using Deepgram."""

    def __init__(self):
        self.dg_client = DeepgramClient(DEEPGRAM_API_KEY)
        self.microphone = None
        self.audio = pyaudio.PyAudio()  # Initialize PyAudio instance

    async def transcribe_speech(self, callback):
        """Transcribes user speech in real time and sends it to callback."""
        options = LiveOptions(model="nova-2", punctuate=True, language="en-US", encoding="linear16", sample_rate=16000)
        while True:
            try:
                dg_connection = self.dg_client.listen.asyncwebsocket.v("1")
                await dg_connection.start(options)
                info = self.audio.get_host_api_info_by_index(0)
                num_devices = info.get('deviceCount')
                for i in range(num_devices):
                    device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
                    if device_info.get('maxInputChannels') > 0:
                        print(f"Input Device id {i} - {device_info.get('name')}")

                input_device_index = 1

                async def on_transcription(self, result, **kwargs):
                    transcript = result.channel.alternatives[0].transcript.strip()
                    if transcript:
                        print(f"User: {transcript}")
                        callback(transcript)

                dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcription)
                self.microphone = Microphone(dg_connection.send, input_device_index=input_device_index)
                self.microphone.start()

                await asyncio.sleep(0.1)  # Adjust duration based on expected speech input
                self.microphone.finish()
                await dg_connection.finish()
            except Exception as e:
                print(f"🔄 Reconnecting to Deepgram WebSocket... Error: {e}")
                self.microphone=None
                await asyncio.sleep(2)

    def speak(self, text):
        """Converts AI-generated text to speech and plays it."""
        url = f"https://api.deepgram.com/v1/speak?model=aura-helios-en&encoding=linear16&sample_rate=24000"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(url, json={"text": text}, headers=headers, stream=True)
        
        audio_file = "response.mp3"
        with open(audio_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        
        os.system("start response.mp3")  # Play audio in Windows

    def __del__(self):  
        """Clean up PyAudio when the object is destroyed."""
        self.audio.terminate()
class AIManager:
    """Manages conversation flow and integrates speech, NLP, and AI chatbot."""

    def __init__(self):
        self.speech = DeepgramSpeech()
        self.chatbot = GeminiChatbot()

    async def start_conversation(self):
        """Begins the voice-based AI conversation."""
        print("🔹 AI Assistant is ready. Speak now!")

        def process_transcription(transcript):
            ai_response = self.chatbot.generate_response(transcript)
            print(f"AI: {ai_response}")
            self.speech.speak(ai_response)

        while True:
            await self.speech.transcribe_speech(process_transcription)
            if SESSION_HISTORY and "goodbye" in SESSION_HISTORY[-1]["user"].lower():
                print("🔹 Ending session. Have a great day!")
                break
if __name__ == "__main__":
    manager = AIManager()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(manager.start_conversation())

    # 🔹 Dashboard Code
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Voice Bot Performance Dashboard"),
        dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
        html.Div(id="total-interactions", style={"fontSize": "20px", "marginBottom": "20px"}),
        dcc.Graph(id="top-questions"),
        dcc.Graph(id="response-time"),
    ])

    @app.callback(
        [Output("total-interactions", "children"),
         Output("top-questions", "figure"),
         Output("response-time", "figure")],
        Input("interval-component", "n_intervals")
    )
    def update_dashboard(n):
        df = pd.read_csv(LOG_FILE)

        total_interactions = f"Total Queries: {len(df)}"
        top_questions = df["user_query"].value_counts().nlargest(5).reset_index()
        fig_questions = px.bar(top_questions, x="index", y="user_query", title="Top Questions")
        fig_response_time = px.histogram(df, x="response_time", nbins=10, title="Response Time")

        return total_interactions, fig_questions, fig_response_time

    app.run_server(debug=True)