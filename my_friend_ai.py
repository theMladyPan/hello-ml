#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech, 
    SpeechT5HifiGan
)
import random
from colors import red, green, yellow, blue, magenta, cyan, white
import logging
import pickle
import os
from datasets import load_dataset
import torch
import soundfile as sf
from playsound import playsound
import tempfile
from threading import Thread


log = logging.getLogger(__name__)
# add timestamp to log
logging.basicConfig(format="%(asctime)s - %(name)s - %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log.info("Logger started")

human = "Stanley"

class Voice:
    _log = logging.getLogger("Voice")
    model_name = "microsoft/speecht5_tts"
    vocoder_name = "microsoft/speecht5_hifigan"
    voice_dataset = "Matthijs/cmu-arctic-xvectors"
    
    def __init__(self):
        self.processor = SpeechT5Processor.from_pretrained(self.model_name)
        self._log.info("Loaded processor")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name)
        self._log.info("Loaded model")
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_name)
        self._log.info("Loaded vocoder")
    
        # load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_dataset(self.voice_dataset, split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)  # 7306 is the speaker id
        self._log.info("Loaded speaker embeddings")
        
    
    def pronounce(self, text: str):
        self._log.debug(f"Going to talk {text}")
        inputs = self.processor(
            text=text, 
            return_tensors="pt"
            )
        self._log.debug("Processing input")
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        self._log.debug("Generated speech")
        return speech

    
    def speak(self, speech):
        # save speech to temporary file and play it
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, speech.numpy(), samplerate=16000)
            self._log.debug("Saved speech")
            playsound(f.name)
            

class Friend:
    _dialog_file_name = "dialog.pkl"
    voice_thread = False
    
    def __init__(self, name: str, large: bool = False):
        self.name = name
        self._log = logging.getLogger(self.name)
        if large: self.model_name = "microsoft/GODEL-v1_1-large-seq2seq"  
        else: self.model_name = "microsoft/GODEL-v1_1-base-seq2seq"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, model_max_length=2048)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.dialog = []
        self.knowledge = f"I am {self.name}, an AI model. {human} is a human, a friend of mine. It is a year 2023 now."
        self.instruction = "Reply in first person"
        self._log.info(f"Loaded model {self.model_name}, loading dialog file...")
        self.load_dialog()
        
        
    def add_voice(self, voice: Voice):
        self.voice = voice
        
    
    def load_dialog(self):
        # check if dialog file exists
        if os.path.isfile(self._dialog_file_name):
            with open(self._dialog_file_name, 'rb') as f:
                self.dialog = pickle.load(f)
            self._log.info(f"Loaded dialog from file {self._dialog_file_name}.")
        else:
            self._log.info(f"Dialog file {self._dialog_file_name} not found. Starting with empty dialog.")
            # create empty dialog file
            with open(self._dialog_file_name, 'wb') as f:
                pickle.dump(self.dialog, f)
            
        
    def save(self):
        # make backup from dialog file in case of crash
        if os.path.isfile(f"{self._dialog_file_name}"):
            with open(self._dialog_file_name, 'rb') as f:
                with open(f"{self._dialog_file_name}.bak", 'wb') as f_bak:
                    f_bak.write(f.read())
            self._log.info(f"Made backup of dialog file {self._dialog_file_name} to {self._dialog_file_name}.bak.")
        
        
        # save dialog to file
        with open(self._dialog_file_name, 'wb') as f:
            pickle.dump(self.dialog, f)           
            self._log.info(f"Saved dialog to file {self._dialog_file_name}.")
            
    
    def think(self, text: str) -> str:
        self._log.debug(f"Thinking about {text}...")
        if text: self.dialog.append(str(text))
        
        if self.knowledge != '':
            knowledge = '[KNOWLEDGE] ' + self.knowledge
        else:
            knowledge = ""
        dialog = ' EOS '.join(self.dialog)
        query = f"{self.instruction} [CONTEXT] {dialog} {knowledge}"
        input_ids = self.tokenizer(f"{query}", return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids, 
            max_length=128, 
            min_length=2, 
            top_p=.95, 
            do_sample=True, 
            temperature=1.2, 
            # num_return_sequences=random.choice([1, 1, 2, 3, 5]),
            early_stopping=True,
            no_repeat_ngram_size=3,
            # num_beams=9,
            length_penalty=-1.0,
            repetition_penalty=1.5
        )

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self._log.info(green(f"{output}"))
        self.dialog.append(output)
        
        return output
    
    
    def speak(self, speech) -> None:
        # wait while last sentence is being spoken
        if self.voice_thread:
            self.voice_thread.join()
        # speak in a new thread to avoid blocking
        self.voice_thread = Thread(target=self.voice.speak, args=(speech,))
        self.voice_thread.start()
    
    
    def reply(self, text: str) -> str:
        reply = self.think(text)
        speech = self.voice.pronounce(reply)
        self.speak(speech)
        return reply
            

if __name__ == "__main__":
    f = Friend("Jillian", large=True)
    f_voice = Voice()
    f.add_voice(f_voice)
    f_voice.speak(f_voice.pronounce("Hello there!"))
    
    text = ""
    while(True):
        try:
            f.reply(text=input("> "))
        except KeyboardInterrupt:
            f.save()
            break
        except Exception as e:
            f.save()
            raise
    
    log.info("Bye bye!")
    