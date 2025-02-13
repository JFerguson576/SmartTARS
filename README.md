#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TARSOS60.py

An advanced robotic assistant integrating voice control, face recognition, image classification,
sentiment analysis, and a comprehensive dashboard for real-time monitoring and interaction.
Includes improvements to voice interactions, camera warning handling, GPIO error management,
conversation memory, personality switching, and continuous recognition.

**Heart Rate module has been fully removed.**
"""

import os
import sys
import re
import json
import glob
import sqlite3
import logging
import traceback
import datetime
import time
import random
import threading
import queue
import struct
from collections import deque
from threading import Lock

import tkinter as tk
from tkinter import ttk, scrolledtext

import cv2
import numpy as np
import requests
import pygame
from PIL import Image, ImageTk

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

import pvporcupine
import pyaudio
import speech_recognition as sr
from vosk import Model as VoskModel, KaldiRecognizer

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import OpenAI for GPT interaction
import openai

# Import PyPDF2 for PDF reading
import PyPDF2

# Import Sumy for text summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Initialize Pygame mixer for audio playback
pygame.mixer.init()

# Load environment variables
load_dotenv(dotenv_path='/home/tars/TARSOSenv/.env')

# --- Constants and Configuration ---
LOG_FILE = '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/logs/tars.log'
DATABASE_PATH = '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/database/tars.db'
IMAGE_PATH = '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/images'
STARTUP_SOUNDS_PATH = '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/startup_sounds'
FACTS_FOLDER_PATH = '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/facts'
MEMORIES_FOLDER_PATH = '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/memories'
DOCUMENTS_FOLDER_PATH = '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/user_documents'
KNOWN_FACES_PATH = '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/known_faceses'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')
ELEVEN_LABS_VOICE_ID = os.getenv('ELEVEN_LABS_VOICE_ID')
PICOVOICE_ACCESS_KEY = os.getenv('PICOVOICE_ACCESS_KEY')
PICOVOICE_WAKE_WORD_PATH = os.getenv("PICOVOICE_WAKE_WORD_PATH", "/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/porcupine/hey-TARS_en_raspberry-pi_v3_0_0.ppn")
VOSK_MODEL_PATH = os.getenv('VOSK_MODEL_PATH', '/media/tars/6cf4e983-6b32-40d5-9f01-a770df7bdd0a/TARS_memory/models/vosk-model-en-us-0.22/am/final.mdl')

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize spaCy for NLU
nlp = spacy.load("en_core_web_sm")

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Check GPIO Availability (Assuming Raspberry Pi)
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO_AVAILABLE = True
    logging.debug("RPi.GPIO imported successfully.")
except ImportError:
    logging.warning("RPi.GPIO not available. GPIO functionalities will be disabled.")
except RuntimeError as e:
    logging.error(f"RuntimeError while importing RPi.GPIO: {e}")
    GPIO_AVAILABLE = False

# Define GPIO Pins for LEDs
POWER_LED_PIN = 17
LISTEN_LED_PIN = 27
JOKE_LED_PIN = 22

LED_ERROR_OCCURRED = False  # Flag to prevent repeated LED errors

def led_on(pin):
    """Turn on the specified LED if GPIO is available."""
    global LED_ERROR_OCCURRED
    if GPIO_AVAILABLE and (not LED_ERROR_OCCURRED):
        try:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)
            logging.info(f"LED on pin {pin} turned ON.")
        except RuntimeError as e:
            logging.error(f"Error turning on LED on pin {pin}: {e}")
            logging.error(traceback.format_exc())
            LED_ERROR_OCCURRED = True
    else:
        logging.debug(f"Skipping led_on for pin {pin}; GPIO unavailable or error occurred.")

def led_off(pin):
    """Turn off the specified LED if GPIO is available."""
    global LED_ERROR_OCCURRED
    if GPIO_AVAILABLE and (not LED_ERROR_OCCURRED):
        try:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
            logging.info(f"LED on pin {pin} turned OFF.")
        except RuntimeError as e:
            logging.error(f"Error turning off LED on pin {pin}: {e}")
            logging.error(traceback.format_exc())
            LED_ERROR_OCCURRED = True
    else:
        logging.debug(f"Skipping led_off for pin {pin}; GPIO unavailable or error occurred.")

def turn_off_all_leds():
    """Turn off all LEDs safely."""
    if GPIO_AVAILABLE and (not LED_ERROR_OCCURRED):
        try:
            GPIO.setup(POWER_LED_PIN, GPIO.OUT)
            GPIO.setup(LISTEN_LED_PIN, GPIO.OUT)
            GPIO.setup(JOKE_LED_PIN, GPIO.OUT)
            GPIO.output(POWER_LED_PIN, GPIO.LOW)
            GPIO.output(LISTEN_LED_PIN, GPIO.LOW)
            GPIO.output(JOKE_LED_PIN, GPIO.LOW)
            logging.info("All LEDs turned OFF.")
        except RuntimeError as e:
            logging.error(f"Error turning off all LEDs: {e}")
            logging.error(traceback.format_exc())

power_led = POWER_LED_PIN
listen_led = LISTEN_LED_PIN
joke_led = JOKE_LED_PIN

start_time = time.time()
last_activity_time = time.time()

def setup_logging():
    """Configure logging for the application."""
    try:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory at {log_dir}")  
        logging.basicConfig(
            level=logging.DEBUG,  
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging is set up.")
    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1)

def setup_database():
    """Initialize the SQLite database with required tables."""
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            # Create tables if they don't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    tars_response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    duration REAL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    face_encoding BLOB
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognized_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_name TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS personality_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    system_prompt TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER,
                    FOREIGN KEY(profile_id) REFERENCES personality_profiles(id)
                )
            ''')
            conn.commit()
        logging.info("Database setup completed.")
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        logging.error(traceback.format_exc())

class DashboardLogHandler(logging.Handler):
    """Custom logging handler to send logs to the dashboard."""
    def __init__(self, dashboard_app):
        super().__init__()
        self.dashboard_app = dashboard_app
    
    def emit(self, record):
        log_entry = self.format(record)
        if "Failed to capture frame from camera" in log_entry:
            return  
        self.dashboard_app.log_message(log_entry)

# Camera Module
class CameraModule:
    """Handles camera operations."""
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            logging.error("Camera could not be opened.")
            self.capture = None
        else:
            logging.info("Camera initialized successfully.")
        self.failed_attempts = 0
        self.max_failed_attempts = 5
        self.retry_delay = 1  

    def capture_frame(self):
        """Capture a single frame from the camera with retry mechanism."""
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                self.failed_attempts = 0
                return frame
            else:
                self.failed_attempts += 1
                if self.failed_attempts <= self.max_failed_attempts:
                    logging.warning("Failed to capture frame from camera.")
                elif self.failed_attempts == self.max_failed_attempts + 1:
                    logging.error("Camera capture failed repeatedly. Disabling camera module.")
                    self.capture = None
                time.sleep(self.retry_delay)
                self.retry_delay = min(self.retry_delay * 2, 10) 
                return None
        else:
            return None

    def release(self):
        """Release the camera resource."""
        if self.capture:
            self.capture.release()
            logging.info("Camera released.")

# Face Recognition Module
class FaceRecognitionModule:
    """Handles face recognition using camera frames."""
    def __init__(self, camera):
        self.camera = camera
        self.running = False
        self.thread = None
        self.current_user_name = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logging.info("FaceRecognitionModule initialized.")
    
    def start(self):
        """Start face recognition."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.recognition_loop, daemon=True)
            self.thread.start()
            logging.info("Face recognition started.")
    
    def stop(self):
        """Stop face recognition."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            logging.info("Face recognition stopped.")
    
    def recognition_loop(self):
        """Continuously perform face recognition."""
        while self.running:
            if self.camera is None or self.camera.capture is None:
                time.sleep(2)
                continue

            frame = self.camera.capture_frame()
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    self.current_user_name = "Captain"
                    logging.info(f"Recognized user: {self.current_user_name}")
                    self.stop()
            time.sleep(1)

# Image Classification Module
class ImageClassificationModule:
    """Handles image classification using TensorFlow Lite models."""
    def __init__(self, model_path, label_path, camera, listen_led, use_tpu=False, dashboard_app=None):
        self.model_path = model_path
        self.label_path = label_path
        self.camera = camera
        self.use_tpu = use_tpu
        self.dashboard_app = dashboard_app
        self.running = False
        self.thread = None
        self.labels = self.load_labels()
        self.interpreter = self.load_model()
        if self.interpreter:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.height = self.input_details[0]['shape'][1]
            self.width = self.input_details[0]['shape'][2]
            logging.info("ImageClassificationModule initialized successfully.")
        else:
            logging.error("Failed to load TFLite model. ImageClassificationModule will not run.")
    
    def load_labels(self):
        """Load labels from the label file and remove label numbers."""
        try:
            with open(self.label_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            processed_labels = []
            for label in labels:
                label = re.sub(r'^\d+\s+', '', label)
                processed_labels.append(label)
            logging.info(f"Loaded {len(processed_labels)} labels for image classification.")
            return processed_labels
        except Exception as e:
            logging.error(f"Error loading labels: {e}")
            return []
    
    def load_model(self):
        """Load the TFLite model, with Edge TPU support if available."""
        try:
            from tflite_runtime.interpreter import Interpreter
            from tflite_runtime.interpreter import load_delegate
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter, load_delegate
            except ImportError:
                logging.error("Failed to import TensorFlow Lite Interpreter.")
                return None
    
        try:
            if self.use_tpu:
                interpreter = Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[load_delegate('libedgetpu.so.1')]
                )
                logging.info("Loaded TFLite model with Edge TPU delegate.")
            else:
                interpreter = Interpreter(model_path=self.model_path)
                logging.info("Loaded TFLite model without Edge TPU.")
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            logging.error(f"Error loading TFLite model: {e}")
            return None
    
    def announce_and_log(self, label, confidence, user_id=None):
        """Announce the detected object and log it."""
        if label == "Unknown":
            logging.info("Object recognized but label is unknown. Not announcing.")
            return
        message = f"I see a {label}."
        logging.info(message)
        speak(message)
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO recognized_objects (object_name, confidence, timestamp) VALUES (?, ?, ?)',
                    (label, confidence, datetime.datetime.now())
                )
            logging.info(f"Logged recognized object: {label} ({confidence*100:.1f}%)")
        except Exception as e:
            logging.error(f"Error logging recognized object: {e}")
            logging.error(traceback.format_exc())
        if self.dashboard_app:
            self.dashboard_app.update_recognized_objects(
                [{"object_name": label, "confidence": confidence}]
            )
            self.dashboard_app.log_message(f"TARS: {message}")
    
    def start(self):
        """Start the image classification thread."""
        if self.interpreter is None:
            logging.error("Interpreter not initialized. Cannot start image classification.")
            return
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.classification_loop, daemon=True)
            self.thread.start()
            logging.info("Image Classification started.")
    
    def stop(self):
        """Stop the image classification thread."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            logging.info("Image Classification stopped.")
    
    def classification_loop(self):
        """Continuously perform image classification."""
        if self.interpreter is None:
            logging.error("Interpreter not initialized. Exiting classification loop.")
            return
        logging.info("Image classification loop started.")
        while self.running:
            if self.camera is None or self.camera.capture is None:
                time.sleep(2)
                continue

            frame = self.camera.capture_frame()
            if frame is None:
                time.sleep(0.5)
                continue

            try:
                input_data = cv2.resize(frame, (self.width, self.height))
                input_data = np.expand_dims(input_data, axis=0)
                input_data = (np.float32(input_data) - 127.5) / 127.5  
            except Exception as e:
                logging.error(f"Error preprocessing frame: {e}")
                continue

            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
            except Exception as e:
                logging.error(f"Error during model inference: {e}")
                continue

            try:
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            except Exception as e:
                logging.error(f"Error retrieving model output: {e}")
                continue

            try:
                adjusted_output_data = output_data
                top_k = adjusted_output_data.argsort()[-5:][::-1]
                for idx in top_k:
                    confidence = adjusted_output_data[idx]
                    if idx < len(self.labels):
                        label = self.labels[idx]
                    else:
                        label = "Unknown"
                    logging.info(f"Image Classification: {label} ({confidence*100:.2f}%)")
                    if confidence > 0.70 and label != "Unknown":
                        self.announce_and_log(label, confidence)
                        break
            except Exception as e:
                logging.error(f"Error processing model output: {e}")
                continue

            time.sleep(0.5)

# UPS Power Module (Placeholder)
class UPSPowerModule:
    """Handles UPS power status."""
    def __init__(self, addr=0x41):
        self.addr = addr
        self.status = "Operational"
        logging.info("UPSPowerModule initialized.")
    
    def start(self):
        """Start monitoring UPS status."""
        logging.info("UPSPowerModule started.")
    
    def stop(self):
        """Stop monitoring UPS status."""
        logging.info("UPSPowerModule stopped.")

###############################################################################
#                           Voice Control Module
###############################################################################
class VoiceControlModule:
    """
    Handles voice commands using Porcupine for wake word detection.
    Also supports partial/continuous recognition using Vosk if desired.
    """
    LISTENING_FOR_WAKE_WORD = 0
    CONVERSATION = 1

    def __init__(self, command_callback, listen_led, use_vosk=False, use_partial_recognition=False):
        self.command_callback = command_callback
        self.listen_led = listen_led
        self.running = False
        self.thread = threading.Thread(target=self.listen_for_commands, daemon=True)
        self.recognizer = sr.Recognizer()
        self.conversation_state = self.LISTENING_FOR_WAKE_WORD
        self.conversation_timeout = 10
        self.last_command_time = 0

        self.use_vosk = use_vosk
        self.use_partial_recognition = use_partial_recognition

        try:
            self.microphone = sr.Microphone()
        except Exception as e:
            logging.error(f"Error initializing microphone: {e}")
            self.microphone = None
        
        if self.use_vosk:
            self.load_vosk_model()

        access_key = PICOVOICE_ACCESS_KEY
        wake_word_path = PICOVOICE_WAKE_WORD_PATH
        if not access_key or not wake_word_path:
            logging.error("Porcupine wake word detection is not fully configured.")
            self.porcupine = None
            return

        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[wake_word_path]
            )
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            logging.info("Porcupine initialized for 'Hey TARS' wake word.")
        except Exception as e:
            logging.error(f"Failed to initialize Porcupine: {e}")
            self.porcupine = None

        self.cooldown = 3
        self.last_activation = 0

    def load_vosk_model(self):
        """Load the Vosk model for offline speech recognition."""
        global vosk_model, recognizer_vosk
        if not os.path.exists(VOSK_MODEL_PATH):
            logging.error(f"Vosk model not found at {VOSK_MODEL_PATH}.")
            return
        vosk_model = VoskModel(VOSK_MODEL_PATH)
        recognizer_vosk = KaldiRecognizer(vosk_model, 16000)
        logging.info("Vosk model loaded for offline speech recognition.")

    def start(self):
        """Start the voice control thread."""
        if not self.running and self.porcupine is not None and self.microphone is not None:
            self.running = True
            self.thread.start()
            logging.info("Voice Control Module started.")

    def stop(self):
        """Stop the voice control thread."""
        if self.running:
            self.running = False
            self.thread.join()
            logging.info("Voice Control Module stopped.")

    def listen_for_commands(self):
        if self.microphone is None or self.porcupine is None:
            logging.error("Voice Control cannot start without microphone or Porcupine.")
            return

        logging.info("Voice Control: Listening for wake word or conversation...")

        try:
            while self.running:
                if self.conversation_state == self.LISTENING_FOR_WAKE_WORD:
                    pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                    pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                    pcm = np.array(pcm, dtype=np.int16)

                    result = self.porcupine.process(pcm)
                    if result >= 0:
                        current_time = time.time()
                        if current_time - self.last_activation < self.cooldown:
                            logging.info("Wake word detected but in cooldown. Ignoring.")
                            continue
                        logging.info("Wake word detected. Speaking 'Yes?'")
                        speak("Yes?", add_filler=False)
                        led_on(self.listen_led)
                        self.last_activation = current_time
                        self.last_command_time = time.time()
                        self.conversation_state = self.CONVERSATION

                elif self.conversation_state == self.CONVERSATION:
                    if self.use_vosk and self.use_partial_recognition:
                        self.handle_continuous_recognition()
                    else:
                        self.handle_single_phrase_recognition()

                    if time.time() - self.last_command_time > self.conversation_timeout:
                        logging.info("Conversation timeout reached, returning to wake word mode.")
                        led_off(self.listen_led)
                        self.conversation_state = self.LISTENING_FOR_WAKE_WORD
        finally:
            if hasattr(self, 'audio_stream'):
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if hasattr(self, 'pa'):
                self.pa.terminate()
            if hasattr(self, 'porcupine'):
                self.porcupine.delete()
            logging.info("Voice Control Module stopped listening loop.")

    def handle_single_phrase_recognition(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            logging.debug("Conversation Mode (single phrase): Listening...")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=2,
                    phrase_time_limit=8
                )
            except sr.WaitTimeoutError:
                return

        led_off(self.listen_led)
        try:
            if self.use_vosk and 'recognizer_vosk' in globals():
                command = self.recognize_with_vosk(audio)
            else:
                command = self.recognizer.recognize_google(audio)
            logging.info(f"Conversation Mode (single phrase): Command: {command}")
            self.handle_command_text(command)
        except sr.UnknownValueError:
            logging.info("Did not understand audio.")
            speak("Sorry, could you repeat that?")
        except sr.RequestError as e:
            logging.error(f"Could not request results: {e}")
            speak("I'm sorry, I couldn't process that.")
        except Exception as e:
            logging.error(f"Error in conversation mode: {e}")
            speak("I'm sorry, something went wrong.")
        finally:
            self.last_command_time = time.time()
            led_on(self.listen_led)

    def handle_continuous_recognition(self):
        logging.debug("Conversation Mode (continuous partial recognition).")
        partial_command_buffer = ""
        start_listen_time = time.time()
        while time.time() - start_listen_time < self.conversation_timeout:
            data = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, data)
            pcm = np.array(pcm, dtype=np.int16)
            audio_data = pcm.tobytes()

            if recognizer_vosk.AcceptWaveform(audio_data):
                result_json = recognizer_vosk.Result()
                result_text = json.loads(result_json)["text"]
                if result_text.strip():
                    logging.info(f"[Final] recognized text: {result_text}")
                    self.handle_command_text(result_text)
                    self.last_command_time = time.time()
                    return
            else:
                partial_json = recognizer_vosk.PartialResult()
                partial_text = json.loads(partial_json)["partial"]
                if partial_text:
                    logging.debug(f"[Partial] recognized text: {partial_text}")
                    if "stop" in partial_text.lower() or "cancel" in partial_text.lower():
                        speak("Command canceled.")
                        self.last_command_time = time.time()
                        return
                    partial_command_buffer += " " + partial_text
            time.sleep(0.05)

        final_cmd = partial_command_buffer.strip()
        if final_cmd:
            logging.info(f"Timeout final command: {final_cmd}")
            self.handle_command_text(final_cmd)
        self.last_command_time = time.time()

    def handle_command_text(self, command):
        if re.search(r'switch to (\w+) mode', command, re.IGNORECASE):
            match = re.search(r'switch to (\w+) mode', command, re.IGNORECASE)
            if match:
                profile_name = match.group(1).capitalize()
                self.command_callback(f"Switch to {profile_name} mode")
                speak(f"Switching to {profile_name} mode.")
                return
        self.command_callback(command)

    def recognize_with_vosk(self, audio):
        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        if recognizer_vosk.AcceptWaveform(audio_data):
            result = recognizer_vosk.Result()
            command = json.loads(result)["text"]
        else:
            result = recognizer_vosk.PartialResult()
            command = json.loads(result)["partial"]
        return command

###############################################################################
#                          Facts, Memories, Documents
###############################################################################
class FactsManager(FileSystemEventHandler):
    """Monitors a folder for new fact files and updates the database."""

    def __init__(self, facts_folder):
        self.facts_folder = facts_folder
        os.makedirs(self.facts_folder, exist_ok=True)
        self.lock = Lock()
        self.observer = Observer()
        self.observer.schedule(self, self.facts_folder, recursive=False)
        self.observer.start()
        self.load_existing_facts()
        logging.info("FactsManager initialized and observer started.")

    def load_existing_facts(self):
        try:
            for filename in os.listdir(self.facts_folder):
                file_path = os.path.join(self.facts_folder, filename)
                self.process_file(file_path)
        except Exception as e:
            logging.error(f"Error loading existing facts: {e}")

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            logging.info(f"New fact file detected: {event.src_path}")
            self.process_file(event.src_path)

    def process_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            if content:
                user_id = None
                with self.lock:
                    with sqlite3.connect(DATABASE_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute('INSERT INTO facts (user_id, content) VALUES (?, ?)', (user_id, content))
                logging.info(f"Fact added: {file_path}")
            else:
                logging.debug(f"Fact file is empty: {file_path}")
        except Exception as e:
            logging.error(f"Error processing fact file {file_path}: {e}")

    def get_facts_for_user(self, user_id):
        try:
            with self.lock:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    if user_id:
                        cursor.execute('SELECT content FROM facts WHERE user_id = ?', (user_id,))
                    else:
                        cursor.execute('SELECT content FROM facts')
                    facts = cursor.fetchall()
            return [fact[0] for fact in facts]
        except Exception as e:
            logging.error(f"Error retrieving facts: {e}")
            return []

class MemoriesManager:
    """Manages memories added by the user and ensures persistence across sessions."""

    def __init__(self, memories_folder):
        self.memories_folder = memories_folder
        os.makedirs(self.memories_folder, exist_ok=True)
        self.lock = Lock()
        self.load_existing_memories()
        logging.info("MemoriesManager initialized.")

    def load_existing_memories(self):
        try:
            with self.lock:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT content FROM memories ORDER BY timestamp ASC')
                    memories = cursor.fetchall()
            self.memories = [m[0] for m in memories]
            logging.info(f"Loaded {len(self.memories)} memories from DB.")
        except Exception as e:
            logging.error(f"Error loading memories: {e}")
            self.memories = []

    def add_memory(self, content):
        try:
            with self.lock:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO memories (content) VALUES (?)', (content,))
            self.memories.append(content)
            logging.info(f"New memory added: {content}")
        except Exception as e:
            logging.error(f"Error adding memory: {e}")

    def get_all_memories(self):
        return self.memories

class DocumentManager:
    """Manages documents in TARS_memory and extracts useful information."""

    def __init__(self, documents_folder):
        self.documents_folder = documents_folder
        os.makedirs(self.documents_folder, exist_ok=True)
        self.documents = {}
        self.lock = Lock()
        self.load_documents()
        logging.info("DocumentManager initialized.")

    def load_documents(self):
        try:
            file_paths = glob.glob(os.path.join(self.documents_folder, '*'))
            for file_path in file_paths:
                if os.path.isfile(file_path):
                    if file_path in self.documents:
                        continue
                    content = None
                    if file_path.lower().endswith('.pdf'):
                        content = self.read_pdf(file_path)
                    else:
                        content = self.read_file_with_encodings(file_path)
                    if content is None:
                        logging.error(f"Skipping file: {file_path}")
                        continue
                    summary = self.summarize_content(content)
                    with self.lock:
                        self.documents[file_path] = summary
            logging.info("Documents loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading documents: {e}")

    def read_file_with_encodings(self, file_path):
        encodings = ['utf-8', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logging.info(f"Read file {file_path} with encoding {encoding}")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                return None
        logging.error(f"Unable to read file {file_path} with known encodings.")
        return None

    def read_pdf(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                content = ''
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text
            logging.info(f"Read PDF file {file_path}")
            return content
        except Exception as e:
            logging.error(f"Error reading PDF: {e}")
            return None

    def summarize_content(self, content):
        try:
            parser = PlaintextParser.from_string(content, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            summary_sentences = summarizer(parser.document, sentences_count=3)
            summary = ' '.join(str(sentence) for sentence in summary_sentences)
            return summary if summary else content
        except Exception as e:
            logging.error(f"Error summarizing content: {e}")
            return content

    def get_relevant_documents(self, query, max_documents=3):
        try:
            from nltk.tokenize import sent_tokenize
            query_sentences = sent_tokenize(query)
            query_words = set(word.lower() for sentence in query_sentences for word in sentence.split())
            relevant_docs = []
            with self.lock:
                for summary in self.documents.values():
                    summary_sentences = sent_tokenize(summary)
                    summary_words = set(word.lower() for sentence in summary_sentences for word in sentence.split())
                    if query_words & summary_words:
                        relevant_docs.append(summary)
                    if len(relevant_docs) >= max_documents:
                        break
            return "\n".join(relevant_docs)
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return ""

# The Tkinter Dashboard
class Dashboard(tk.Frame):
    """Tkinter-based dashboard showing conversation logs, recognized objects, system status, etc."""

    def __init__(self, master=None, ups_module=None):
        super().__init__(master)
        self.master = master
        if self.master:
            self.master.title("TARS Dashboard")
            self.master.geometry("1200x800")
            self.master.configure(bg='black')
        self.pack(fill='both', expand=True)
        self.ups_module = ups_module

        self.log_queue = queue.Queue()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        self.personality_profiles = {
            "Default": (
                "You are TARS, a witty and helpful robot assistant. "
                "TARS is an advanced robot with a dry sense of humor and unwavering loyalty to the users. "
                "TARS is practical and has problem-solving skills. "
                "TARS responses can be quirky and sometimes sarcastic. "
                "Make responses humorous, and sometimes with irony and wit."
            ),
            "Comedic": (
                "You are TARS, a humorous and entertaining robot assistant. "
                "TARS loves to tell jokes and make witty remarks while assisting the user. "
                "Maintain a light-hearted and fun demeanor in all interactions."
            ),
            "Professional": (
                "You are TARS, a professional and efficient robot assistant. "
                "TARS provides clear, concise, and accurate information. "
                "Maintain a formal and respectful tone."
            )
        }
        self.active_profile = "Default"

        self.create_widgets()
        self.process_log_queue()
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        try:
            self.style = ttk.Style()
            self.style.theme_use('default')
            self.style.configure('TLabel', background='black', foreground='#ADD8E6')
            self.style.configure('TFrame', background='black')
            self.style.configure('TLabelframe', background='black', foreground='#ADD8E6')
            self.style.configure('TLabelframe.Label', background='black', foreground='#ADD8E6')
            self.style.configure('TButton', background='black', foreground='#ADD8E6')
            self.style.configure('TEntry', foreground='black')

            self.notebook = ttk.Notebook(self)
            self.notebook.pack(pady=10, padx=10, fill='both', expand=True)

            # Conversation Tab
            conversation_tab = ttk.Frame(self.notebook, style='TFrame')
            conversation_tab.grid_rowconfigure(0, weight=1)
            conversation_tab.grid_columnconfigure(0, weight=1)
            self.notebook.add(conversation_tab, text='Conversation')

            self.conversation_display = scrolledtext.ScrolledText(
                conversation_tab,
                state='disabled',
                bg='black',
                fg='#ADD8E6',
                font=("Helvetica", 10),
                wrap='word'
            )
            self.conversation_display.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=5)

            # Recognized Objects Tab
            objects_tab = ttk.Frame(self.notebook, style='TFrame')
            objects_tab.grid_rowconfigure(0, weight=1)
            objects_tab.grid_columnconfigure(0, weight=1)
            self.notebook.add(objects_tab, text='Recognized Objects')

            self.objects_listbox = tk.Listbox(objects_tab, bg='black', fg='#ADD8E6', font=("Helvetica", 10))
            self.objects_listbox.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=5)
            objects_scrollbar = ttk.Scrollbar(objects_tab, command=self.objects_listbox.yview)
            objects_scrollbar.grid(row=0, column=1, sticky='ns', padx=(5, 0), pady=5)
            self.objects_listbox['yscrollcommand'] = objects_scrollbar.set

            # System Status Tab
            system_status_tab = ttk.Frame(self.notebook, style='TFrame')
            system_status_tab.grid_rowconfigure(0, weight=1)
            system_status_tab.grid_columnconfigure(0, weight=1)
            self.notebook.add(system_status_tab, text='System Status')

            self.system_status_var = tk.StringVar(value="Operational")
            system_status_label = ttk.Label(system_status_tab, textvariable=self.system_status_var, style='TLabel', font=("Helvetica", 16))
            system_status_label.pack(pady=20)

            # Feedback Tab
            feedback_tab = ttk.Frame(self.notebook, style='TFrame')
            feedback_tab.grid_rowconfigure(0, weight=1)
            feedback_tab.grid_columnconfigure(0, weight=1)
            self.notebook.add(feedback_tab, text='Feedback')

            feedback_frame = ttk.LabelFrame(feedback_tab, text="Submit Feedback", style='TLabelframe')
            feedback_frame.pack(pady=10, padx=10, fill='x')

            self.feedback_entry = ttk.Entry(feedback_frame, width=50)
            self.feedback_entry.pack(side='left', padx=(10, 5), pady=5)

            feedback_button = ttk.Button(feedback_frame, text="Submit", command=self.submit_feedback)
            feedback_button.pack(side='left', padx=(5, 10))

            # Personality Settings Tab
            personality_tab = ttk.Frame(self.notebook, style='TFrame')
            personality_tab.grid_rowconfigure(0, weight=1)
            personality_tab.grid_columnconfigure(0, weight=1)
            self.notebook.add(personality_tab, text='Personality Settings')

            personality_frame = ttk.LabelFrame(personality_tab, text="Manage Personality Profiles", style='TLabelframe')
            personality_frame.pack(pady=10, padx=10, fill='both', expand=True)

            self.profiles_listbox = tk.Listbox(personality_frame, bg='black', fg='#ADD8E6', font=("Helvetica", 10))
            self.profiles_listbox.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=10)
            self.populate_profiles()

            profiles_scrollbar = ttk.Scrollbar(personality_frame, command=self.profiles_listbox.yview)
            profiles_scrollbar.pack(side='left', fill='y', padx=(0, 10), pady=10)
            self.profiles_listbox['yscrollcommand'] = profiles_scrollbar.set

            profiles_buttons_frame = ttk.Frame(personality_frame, style='TFrame')
            profiles_buttons_frame.pack(side='left', fill='y', padx=(0, 10), pady=10)

            add_profile_button = ttk.Button(profiles_buttons_frame, text="Add Profile", command=self.add_profile)
            add_profile_button.pack(pady=(0, 10))

            remove_profile_button = ttk.Button(profiles_buttons_frame, text="Remove Profile", command=self.remove_profile)
            remove_profile_button.pack(pady=(0, 10))

            switch_profile_button = ttk.Button(profiles_buttons_frame, text="Switch Profile", command=self.switch_profile)
            switch_profile_button.pack(pady=(0, 10))

            self.active_profile_var = tk.StringVar(value=f"Active Profile: {self.active_profile}")
            active_profile_label = ttk.Label(personality_frame, textvariable=self.active_profile_var, style='TLabel', font=("Helvetica", 12))
            active_profile_label.pack(pady=(10, 0))

            # Accessibility Tab
            accessibility_frame = ttk.LabelFrame(self.notebook, text="Accessibility", style='TLabelframe')
            accessibility_frame.grid_rowconfigure(0, weight=1)
            accessibility_frame.grid_columnconfigure(0, weight=1)
            self.notebook.add(accessibility_frame, text='Accessibility')

            self.high_contrast_var = tk.BooleanVar()
            high_contrast_check = ttk.Checkbutton(
                accessibility_frame,
                text="Enable High Contrast",
                variable=self.high_contrast_var,
                command=self.toggle_high_contrast
            )
            high_contrast_check.pack(pady=10, padx=10, anchor='w')

            font_size_frame = ttk.LabelFrame(accessibility_frame, text="Font Size", style='TLabelframe')
            font_size_frame.pack(pady=10, padx=10, fill='x')

            increase_font_button = ttk.Button(font_size_frame, text="Increase", command=self.increase_font_size)
            increase_font_button.pack(side='left', padx=(10, 5), pady=5)

            decrease_font_button = ttk.Button(font_size_frame, text="Decrease", command=self.decrease_font_size)
            decrease_font_button.pack(side='left', padx=(5, 10), pady=5)

            shortcuts_label = ttk.Label(accessibility_frame, text="Keyboard Shortcuts:\nCtrl+N: Next Tab\nCtrl+P: Previous Tab", style='TLabel')
            shortcuts_label.pack(pady=10, padx=10, anchor='w')

            self.master.bind('<Control-n>', self.next_tab)
            self.master.bind('<Control-p>', self.previous_tab)
        except Exception as e:
            logging.error(f"Error creating widgets: {e}")
            logging.error(traceback.format_exc())

    def populate_profiles(self):
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name FROM personality_profiles')
                profiles = cursor.fetchall()
            self.profiles_listbox.delete(0, tk.END)
            for profile in profiles:
                self.profiles_listbox.insert(tk.END, profile[0])
        except Exception as e:
            logging.error(f"Error populating profiles: {e}")

    def add_profile(self):
        def save_profile():
            profile_name = entry.get().strip()
            system_prompt = text.get("1.0", tk.END).strip()
            if profile_name and system_prompt:
                try:
                    with sqlite3.connect(DATABASE_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute('INSERT INTO personality_profiles (name, system_prompt) VALUES (?, ?)', (profile_name, system_prompt))
                        conn.commit()
                    self.profiles_listbox.insert(tk.END, profile_name)
                    add_window.destroy()
                    logging.info(f"Added new profile: {profile_name}")
                except sqlite3.IntegrityError:
                    error_label.config(text="Profile name already exists.")
                except Exception as e:
                    logging.error(f"Error adding profile: {e}")
                    error_label.config(text="Failed to add profile.")
            else:
                error_label.config(text="All fields are required.")

        add_window = tk.Toplevel(self.master)
        add_window.title("Add Personality Profile")
        add_window.geometry("400x300")
        add_window.configure(bg='black')

        ttk.Label(add_window, text="Profile Name:", style='TLabel').pack(pady=(10, 0))
        entry = ttk.Entry(add_window, width=50)
        entry.pack(pady=5, padx=10)

        ttk.Label(add_window, text="System Prompt:", style='TLabel').pack(pady=(10, 0))
        text = scrolledtext.ScrolledText(add_window, width=50, height=10, wrap='word')
        text.pack(pady=5, padx=10)

        error_label = ttk.Label(add_window, text="", foreground="red", background="black")
        error_label.pack(pady=5)

        save_button = ttk.Button(add_window, text="Save", command=save_profile)
        save_button.pack(pady=10)

    def remove_profile(self):
        selected = self.profiles_listbox.curselection()
        if selected:
            profile_name = self.profiles_listbox.get(selected[0])
            if profile_name == self.active_profile:
                speak("Cannot remove the active profile. Please switch to another profile first.")
                return
            try:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM personality_profiles WHERE name = ?', (profile_name,))
                    conn.commit()
                self.profiles_listbox.delete(selected[0])
                logging.info(f"Removed profile: {profile_name}")
                speak(f"Profile {profile_name} has been removed.")
            except Exception as e:
                logging.error(f"Error removing profile: {e}")
                speak("Failed to remove the selected profile.")
        else:
            speak("Please select a profile to remove.")

    def switch_profile(self):
        selected = self.profiles_listbox.curselection()
        if selected:
            profile_name = self.profiles_listbox.get(selected[0])
            try:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT system_prompt FROM personality_profiles WHERE name = ?', (profile_name,))
                    result = cursor.fetchone()
                    if result:
                        cursor.execute('DELETE FROM active_profile')
                        cursor.execute('INSERT INTO active_profile (profile_id) VALUES ((SELECT id FROM personality_profiles WHERE name = ?))', (profile_name,))
                        conn.commit()
                        self.active_profile = profile_name
                        self.active_profile_var.set(f"Active Profile: {self.active_profile}")
                        logging.info(f"Switched to profile: {self.active_profile}")
                        speak(f"Switched to {self.active_profile} mode.")
            except Exception as e:
                logging.error(f"Error switching profile: {e}")
                speak("Failed to switch profile.")
        else:
            speak("Please select a profile to switch.")

    def toggle_high_contrast(self):
        if self.high_contrast_var.get():
            self.style.configure('TLabel', foreground='yellow', background='black')
            self.style.configure('TFrame', background='black')
            self.style.configure('TLabelframe', background='black', foreground='yellow')
            self.style.configure('TLabelframe.Label', background='black', foreground='yellow')
            self.conversation_display.configure(bg='black', fg='yellow')
            self.objects_listbox.configure(bg='black', fg='yellow')
            logging.info("High contrast mode enabled.")
        else:
            self.style.configure('TLabel', foreground='#ADD8E6', background='black')
            self.style.configure('TFrame', background='black')
            self.style.configure('TLabelframe', background='black', foreground='#ADD8E6')
            self.style.configure('TLabelframe.Label', background='black', foreground='#ADD8E6')
            self.conversation_display.configure(bg='black', fg='#ADD8E6')
            self.objects_listbox.configure(bg='black', fg='#ADD8E6')
            logging.info("High contrast mode disabled.")

    def increase_font_size(self):
        try:
            current_font = self.conversation_display['font']
            size = int(current_font.split()[1]) + 2
            self.conversation_display.configure(font=("Helvetica", size))
            self.objects_listbox.configure(font=("Helvetica", size))
            logging.info(f"Font size increased to {size}.")
        except Exception as e:
            logging.error(f"Error increasing font size: {e}")

    def decrease_font_size(self):
        try:
            current_font = self.conversation_display['font']
            size = max(int(current_font.split()[1]) - 2, 8)
            self.conversation_display.configure(font=("Helvetica", size))
            self.objects_listbox.configure(font=("Helvetica", size))
            logging.info(f"Font size decreased to {size}.")
        except Exception as e:
            logging.error(f"Error decreasing font size: {e}")

    def next_tab(self, event):
        current = self.notebook.index("current")
        total = len(self.notebook.tabs())
        if current < total - 1:
            self.notebook.select(current + 1)
            speak("Switched to the next tab.")

    def previous_tab(self, event):
        current = self.notebook.index("current")
        if current > 0:
            self.notebook.select(current - 1)
            speak("Switched to the previous tab.")

    def submit_feedback(self):
        feedback = self.feedback_entry.get().strip()
        if feedback:
            logging.info(f"User Feedback: {feedback}")
            speak("Thank you for your feedback!")
            self.feedback_entry.delete(0, tk.END)
            try:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO user_logs (action) VALUES (?)', (f"Feedback: {feedback}",))
                    conn.commit()
                logging.info("User feedback stored successfully.")
            except Exception as e:
                logging.error(f"Error storing user feedback: {e}")
        else:
            speak("Please enter your feedback before submitting.")

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                self.conversation_display.config(state='normal')
                self.conversation_display.insert(tk.END, f"{message}\n")
                self.conversation_display.see(tk.END)
                self.conversation_display.config(state='disabled')
        except Exception as e:
            logging.error(f"Exception in process_log_queue: {e}")
        finally:
            self.after(100, self.process_log_queue)

    def log_message(self, message):
        self.log_queue.put(message.strip())

    def on_close(self):
        try:
            logging.info("Dashboard window closed.")
            self.master.destroy()
            module_registry.stop_all()
            turn_off_all_leds()
            sys.exit(0)
        except Exception as e:
            logging.error(f"Error during dashboard close: {e}")
            sys.exit(1)

    def get_conversation_log(self):
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT user_input, tars_response, timestamp FROM conversation_log ORDER BY timestamp ASC')
                logs = cursor.fetchall()
            return [{"user_input": log[0], "tars_response": log[1], "timestamp": log[2]} for log in logs]
        except Exception as e:
            logging.error(f"Error getting conversation log: {e}")
            return []

    def get_latest_frame(self):
        return None

    def update_system_status(self, status):
        self.system_status_var.set(status)

    def update_recognized_objects(self, objects):
        self.objects_listbox.delete(0, tk.END)
        for obj in objects:
            obj_text = f"{obj['object_name']} ({obj['confidence']*100:.1f}%)"
            self.objects_listbox.insert(tk.END, obj_text)

# Simple Flask Web Server
from flask import Flask, jsonify

def start_flask_server(dashboard_app):
    app = Flask(__name__)

    @app.route('/api/conversation')
    def api_conversation():
        logs = dashboard_app.get_conversation_log()
        return jsonify({"conversation_log": logs})

    @app.route('/')
    def index():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TARS Dashboard</title>
            <style>
                body { background-color: #000; color: #ADD8E6; font-family: Arial, sans-serif; }
                .section { margin: 20px; }
                ul { list-style-type: none; padding: 0; }
                li { margin-bottom: 10px; }
            </style>
            <script>
                async function fetchData() {
                    const conversation = await fetch('/api/conversation').then(res => res.json());
                    const convoList = document.getElementById('conversation');
                    convoList.innerHTML = '';
                    conversation.conversation_log.forEach(log => {
                        const item = document.createElement('li');
                        item.innerText = `User: ${log.user_input} | TARS: ${log.tars_response} | At: ${log.timestamp}`;
                        convoList.appendChild(item);
                    });
                }
                setInterval(fetchData, 5000);
                window.onload = fetchData;
            </script>
        </head>
        <body>
            <div class="section">
                <h2>Conversation Log</h2>
                <ul id="conversation">
                    <li>N/A</li>
                </ul>
            </div>
        </body>
        </html>
        """

    def run():
        # Change port if 5000 is in use or you want a different port
        app.run(host='0.0.0.0', port=5000)

    flask_thread = threading.Thread(target=run, daemon=True)
    flask_thread.start()
    logging.info("Flask web server started on port 5000.")

# Text-to-Speech / Audio Functions
from nltk.tokenize import sent_tokenize
import nltk
# nltk.download('punkt')

IS_SPEAKING = False
is_speaking_lock = Lock()
display_queue = queue.Queue()
tts_queue = queue.Queue()

last_speech_was_filler = False
last_speech_lock = Lock()

def speak(text, add_filler=True):
    global last_speech_was_filler
    should_add_filler = False

    if add_filler and text.strip().lower() != "yes?":
        with last_speech_lock:
            if not last_speech_was_filler:
                if random.random() < 0.3:
                    should_add_filler = True
                    last_speech_was_filler = True
            else:
                should_add_filler = False
    if should_add_filler:
        filler_words = [
            "Sure, let me see.",
            "One moment, please.",
            "Let me check that for you.",
            "Just a moment.",
            "I'll look into that."
        ]
        filler_text = random.choice(filler_words)
        text = f"{filler_text} {text}"
    else:
        with last_speech_lock:
            last_speech_was_filler = False

    sentences = sent_tokenize(text)
    for sentence in sentences:
        display_queue.put(f"TARS: {sentence}")
        tts_queue.put(sentence)
        time.sleep(0.1)

def speak_filler():
    global last_speech_was_filler
    with last_speech_lock:
        if last_speech_was_filler:
            return
        last_speech_was_filler = True
    filler_phrases = [
        "Sure, let me see.",
        "One moment, please.",
        "Let me check that for you.",
        "Just a moment.",
        "I'll look into that."
    ]
    text = random.choice(filler_phrases)
    for sentence in sent_tokenize(text):
        display_queue.put(f"TARS: {sentence}")
        tts_queue.put(sentence)
        time.sleep(0.1)

def tts_worker():
    global IS_SPEAKING
    while True:
        text = tts_queue.get()
        if text:
            with is_speaking_lock:
                IS_SPEAKING = True
            perform_tts(text)
            with is_speaking_lock:
                IS_SPEAKING = False
        tts_queue.task_done()

def perform_tts(text):
    if not ELEVEN_LABS_API_KEY or not ELEVEN_LABS_VOICE_ID:
        logging.error("Eleven Labs API key or Voice ID not set.")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}/stream"
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.8,
            "similarity_boost": 0.8
        }
    }
    from io import BytesIO
    try:
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=60)
        if response.status_code == 200:
            led_on(joke_led)
            led_off(listen_led)

            audio_buffer = BytesIO()
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    audio_buffer.write(chunk)
            audio_buffer.seek(0)

            try:
                pygame.mixer.music.load(audio_buffer, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error playing audio: {e}")

            led_off(joke_led)
            led_on(listen_led)
            logging.info("Spoken response successfully.")
        else:
            error_message = response.json().get('detail', 'Unknown error')
            logging.error(f"Eleven Labs API error: {response.status_code} - {error_message}")
            led_on(listen_led)
    except Exception as e:
        logging.error(f"Eleven Labs API exception: {e}")
        led_on(listen_led)

def display_worker():
    while True:
        message = display_queue.get()
        if message.startswith("HEARD:") or message.startswith("TARS:"):
            dashboard_app.log_message(message)
        else:
            dashboard_app.log_message(message)
        display_queue.task_done()

def play_startup_sound():
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        sound_files = glob.glob(os.path.join(STARTUP_SOUNDS_PATH, '*.mp3')) + \
                      glob.glob(os.path.join(STARTUP_SOUNDS_PATH, '*.wav'))
        if sound_files:
            sound_file = random.choice(sound_files)
            logging.info(f"Playing startup sound: {sound_file}")
            sound = pygame.mixer.Sound(sound_file)
            sound.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            logging.warning("No startup sound files found.")
    except Exception as e:
        logging.error(f"Error playing startup sound: {e}")

def listen_for_command_blocking(use_vosk=False):
    logging.info("Starting to listen for command.")
    while IS_SPEAKING:
        time.sleep(1)
    led_on(listen_led)
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=30, phrase_time_limit=120)
        if use_vosk and 'recognizer_vosk' in globals():
            command = recognize_with_vosk(audio)
        else:
            command = recognizer.recognize_google(audio)
        global last_activity_time
        last_activity_time = time.time()
        return command
    except sr.WaitTimeoutError:
        logging.warning("Listening timed out.")
        return None
    except sr.UnknownValueError:
        logging.warning("Could not understand the audio.")
        return None
    except sr.RequestError as e:
        logging.error(f"Could not request results: {e}")
        return None
    except Exception as e:
        logging.error(f"Error during speech recognition: {e}")
        return None
    finally:
        led_off(listen_led)

def recognize_with_vosk(audio):
    audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
    if recognizer_vosk.AcceptWaveform(audio_data):
        result = recognizer_vosk.Result()
        command = json.loads(result)["text"]
    else:
        result = recognizer_vosk.PartialResult()
        command = json.loads(result)["partial"]
    return command

###############################################################################
#                  GPT Interaction (DB-based memory)
###############################################################################
def get_db_conversation_history(limit=10):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_input, tars_response
            FROM conversation_log
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
    rows.reverse()

    conversation = []
    for r in rows:
        user_text, tars_text = r
        if user_text:
            conversation.append({"role": "user", "content": user_text})
        if tars_text:
            conversation.append({"role": "assistant", "content": tars_text})
    return conversation

def generate_gpt_response(user_input, active_system_prompt, memories_manager, facts_manager, document_manager, db_history_limit=10):
    if not OPENAI_API_KEY:
        return "I'm sorry, I'm not configured with an OpenAI API key."

    conversation_history = get_db_conversation_history(limit=db_history_limit)
    conversation_history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": active_system_prompt}]

    all_memories = memories_manager.get_all_memories()
    if all_memories:
        memories_context = "\n".join(all_memories)
        messages.append({"role": "system", "content": f"Here are some things you remember:\n{memories_context}"})

    all_facts = facts_manager.get_facts_for_user(None)
    if all_facts:
        facts_context = "\n".join(all_facts)
        messages.append({"role": "system", "content": f"Here are some facts:\n{facts_context}"})

    relevant_docs = document_manager.get_relevant_documents(user_input)
    if relevant_docs:
        messages.append({"role": "system", "content": f"Relevant information:\n{relevant_docs}"})

    messages.extend(conversation_history)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.75,
            stream=False
        )
        response_text = response['choices'][0]['message']['content'].strip()
        return response_text
    except Exception as e:
        logging.error(f"Error calling OpenAI ChatCompletion: {e}")
        return "I'm sorry, I couldn't process that."

def is_joke_or_sarcasm(text):
    keywords = ['joke', 'kidding', 'funny', 'sarcasm', 'sarcastic', 'haha', 'lol']
    return any(word in text.lower() for word in keywords)

def handle_voice_command(command, modules, dashboard_app):
    global last_activity_time
    last_activity_time = time.time()

    speak_filler()

    intent = get_intent(command)
    entities = get_entities(command)

    if intent == "start_module" and "module_name" in entities:
        module_name = entities["module_name"]
        if module_name in modules:
            modules[module_name].start()
            response = f"{module_name.capitalize()} started."
            speak(response)
            dashboard_app.log_message(f"TARS: {response}")
            dashboard_app.update_system_status("Operational")
        else:
            speak(f"I don't recognize the module {module_name}.")
    elif intent == "stop_module" and "module_name" in entities:
        module_name = entities["module_name"]
        if module_name in modules:
            speak(f"Are you sure you want to stop {module_name}?")
            response = listen_for_confirmation()
            if response and response.lower() in ["yes", "yeah", "sure", "confirm"]:
                modules[module_name].stop()
                response_text = f"{module_name.capitalize()} stopped."
                speak(response_text)
                dashboard_app.log_message(f"TARS: {response_text}")
                dashboard_app.update_system_status("Maintenance")
            else:
                speak("Operation cancelled.")
        else:
            speak(f"I don't recognize the module {module_name}.")
    elif intent == "ask_time":
        now = datetime.datetime.now()
        if "time" in command.lower():
            response_text = now.strftime("The current time is %H:%M.")
        else:
            response_text = now.strftime("Today is %A, %B %d, %Y.")
        speak(response_text)
        dashboard_app.log_message(f"TARS: {response_text}")
    else:
        gpt_answer = generate_gpt_response(
            user_input=command,
            active_system_prompt=get_active_system_prompt(),
            memories_manager=memories_manager,
            facts_manager=facts_manager,
            document_manager=document_manager,
            db_history_limit=10
        )
        if not gpt_answer:
            gpt_answer = "I'm sorry, I didn't catch that."

        speak(gpt_answer)
        dashboard_app.log_message(f"TARS: {gpt_answer}")

        if gpt_answer.endswith('?'):
            led_on(listen_led)
        else:
            led_off(listen_led)

        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversation_log (user_input, tars_response, timestamp)
                VALUES (?, ?, ?)
            ''', (command, gpt_answer, datetime.datetime.now()))
            conn.commit()

def listen_for_confirmation():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        response = recognizer.recognize_google(audio)
        return response
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except Exception as e:
        logging.error(f"Error in listen_for_confirmation: {e}")
        return None

def get_active_system_prompt():
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.system_prompt FROM personality_profiles p
            JOIN active_profile a ON p.id = a.profile_id
        ''')
        result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return dashboard_app.personality_profiles["Default"]

def get_intent(command):
    doc = nlp(command.lower())
    for token in doc:
        for intent, keywords in INTENTS.items():
            if token.lemma_ in keywords:
                return intent
    return "unknown"

def get_entities(command):
    doc = nlp(command.lower())
    entities = {}
    for module in MODULES:
        if module in command.lower():
            entities["module_name"] = module
            break
    return entities

INTENTS = {
    "start_module": ["start", "open", "begin"],
    "stop_module": ["stop", "close", "end"],
    "ask_time": ["time", "date", "current time", "current date"]
}

# Removed "heart rate" from MODULES:
MODULES = ["face recognition", "object recognition", "image classification"]

FALLBACK_RESPONSES = [
    "I'm sorry, I didn't quite catch that. Could you please repeat?",
    "Apologies, I didn't understand that. Can you rephrase?",
    "Sorry, I'm not sure I understand. How can I assist you?",
    "I didn't get that. Could you say it differently?"
]

###############################################################################
#                            MAIN LOGIC
###############################################################################
class ModuleRegistry:
    """Simple container for registered modules by name."""
    def __init__(self):
        self.modules = {}

    def register(self, name, module):
        self.modules[name] = module

    def get(self, name):
        return self.modules.get(name)

    def stop_all(self):
        for name, module in self.modules.items():
            stop_func = getattr(module, 'stop', None)
            if callable(stop_func):
                logging.info(f"Stopping module: {name}")
                module.stop()

def get_time_based_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

def log_system_event(event_type):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO system_log (event_type) VALUES (?)", (event_type,))
        conn.commit()

def run_tars_logic(dashboard_app, facts_manager, memories_manager, document_manager, use_vosk=False):
    global start_time
    start_time = time.time()
    log_system_event('start')

    user_name = "Captain"
    greeting = get_time_based_greeting()
    greet_message = f"{greeting}, {user_name}. TARS is operational. How can I assist you today?"
    speak(greet_message, add_filler=False)
    dashboard_app.log_message(f"TARS: {greet_message}")

    if greet_message.endswith('?'):
        led_on(listen_led)

    command_queue = module_registry.get('command_queue')
    while True:
        command = command_queue.get()  
        if command is None:
            break

        remember_match = re.search(r'\b(remember|please\s*remember)(?:\s*that)?\b', command, re.IGNORECASE)
        if remember_match:
            memory_content = command[remember_match.end():].strip()
            if memory_content:
                memories_manager.add_memory(memory_content)
                speak("I've noted that.")
            else:
                speak("I didn't catch that. Could you please repeat?")
        else:
            handle_voice_command(command, module_registry.modules, dashboard_app)

        # Update recognized objects
        recognized_objects = get_recognized_objects()
        dashboard_app.update_recognized_objects(recognized_objects)

        # Update system status from UPS
        system_status = ups_module.status
        dashboard_app.update_system_status(system_status)

def get_recognized_objects():
    recognized = []
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT object_name, confidence FROM recognized_objects ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
        for row in rows:
            recognized.append({"object_name": row[0], "confidence": float(row[1])})
    except Exception as e:
        logging.error(f"Error retrieving recognized objects: {e}")
    return recognized

def video_processing(camera, image_classification_module, voice_control, dashboard_app):
    """
    Handles basic video processing. Heart rate references removed.
    """
    try:
        logging.info("Starting video processing loop.")
        while True:
            if camera is None or camera.capture is None:
                time.sleep(2)
                continue

            frame = camera.capture_frame()
            if frame is None:
                time.sleep(0.5)
                continue
            # If you need future expansions, do so here.

            time.sleep(0.5)
    except Exception as e:
        logging.error(f"Error in video_processing: {e}")

def main():
    global module_registry, GPIO_AVAILABLE, last_activity_time, dashboard_app, start_time
    setup_logging()

    def initialize_pygame():
        try:
            pygame.init()
            pygame.mixer.init()
            logging.info("Pygame initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Pygame: {e}")
            logging.error(traceback.format_exc())

    initialize_pygame()
    setup_database()
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(STARTUP_SOUNDS_PATH, exist_ok=True)
    os.makedirs(MEMORIES_FOLDER_PATH, exist_ok=True)
    os.makedirs(FACTS_FOLDER_PATH, exist_ok=True)
    os.makedirs(DOCUMENTS_FOLDER_PATH, exist_ok=True)

    if GPIO_AVAILABLE:
        try:
            led_on(power_led)
        except Exception as e:
            logging.error(f"Error turning on power LED: {e}")
    else:
        logging.info("GPIO unavailable. LEDs will not be used.")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_vosk', action='store_true', help='Use Vosk for offline speech recognition')
    parser.add_argument('--partial_recognition', action='store_true', help='Use partial/continuous recognition mode')
    args = parser.parse_args()

    module_registry = ModuleRegistry()
    command_queue = queue.Queue(maxsize=50)
    module_registry.register('command_queue', command_queue)

    # Init modules (heart_rate removed)
    camera = CameraModule()
    face_recog = FaceRecognitionModule(camera)
    ups_module = UPSPowerModule()
    image_class_module = ImageClassificationModule(
        model_path='/path/to/your/model.tflite',
        label_path='/path/to/your/labels.txt',
        camera=camera,
        listen_led=listen_led,
        use_tpu=False,
        dashboard_app=None
    )

    module_registry.register('camera', camera)
    module_registry.register('face_recognition', face_recog)
    module_registry.register('ups_module', ups_module)
    module_registry.register('image_classification', image_class_module)

    try:
        root = tk.Tk()
        # Remove heart_rate from Dashboard init
        dashboard_app = Dashboard(root, ups_module=ups_module)
        logging.info("Dashboard initialized.")
    except Exception as e:
        logging.error(f"Failed to init Dashboard: {e}")
        sys.exit(1)

    dashboard_handler = DashboardLogHandler(dashboard_app)
    dashboard_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    dashboard_handler.setFormatter(formatter)
    logging.getLogger().addHandler(dashboard_handler)

    image_class_module.dashboard_app = dashboard_app

    voice_control = VoiceControlModule(
        command_callback=lambda cmd: handle_command_queue(cmd, command_queue),
        listen_led=listen_led,
        use_vosk=args.use_vosk,
        use_partial_recognition=args.partial_recognition
    )
    voice_control.start()
    module_registry.register('voice_control', voice_control)

    face_recog.start()
    image_class_module.start()
    play_startup_sound()

    global facts_manager, memories_manager, document_manager
    facts_manager = FactsManager(FACTS_FOLDER_PATH)
    memories_manager = MemoriesManager(MEMORIES_FOLDER_PATH)
    document_manager = DocumentManager(DOCUMENTS_FOLDER_PATH)

    start_flask_server(dashboard_app)
    last_activity_time = time.time()

    # TARS main logic in a separate thread
    tars_thread = threading.Thread(
        target=run_tars_logic,
        args=(dashboard_app, facts_manager, memories_manager, document_manager, args.use_vosk),
        daemon=True
    )
    tars_thread.start()

    # Video processing thread (no heart_rate)
    video_thread = threading.Thread(
        target=video_processing,
        args=(camera, image_class_module, voice_control, dashboard_app),
        daemon=True
    )
    video_thread.start()

    # Display worker
    display_thread = threading.Thread(target=display_worker, daemon=True)
    display_thread.start()

    # TTS worker
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    try:
        root.mainloop()
    except Exception as e:
        logging.error(f"Exception in mainloop: {e}")
    finally:
        turn_off_all_leds()

def handle_command_queue(command, command_queue):
    logging.info(f"Queueing command: {command}")
    command_queue.put(command)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        sys.exit(1)
