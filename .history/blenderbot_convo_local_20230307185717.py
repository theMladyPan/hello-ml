#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, Conversation, AutoModelForCausalLM
import torch
from pprint import pprint as print
import logging
import pickle

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_name = "facebook/blenderbot_small-90M"  # AutoModelForSeq2SeqLM
model_name = "microsoft/DialoGPT-large"  # AutoModelForCausalLM 1.75GB
model_name = "facebook/blenderbot-1B-distill"  # AutoModelForSeq2SeqLM
model_name = "PygmalionAI/pygmalion-6b"  # AutoModelForCasualLM 
model_name = "facebook/blenderbot-3B"  # AutoModelForSeq2SeqLM
model_name = "facebook/blenderbot-400M-distill"  # AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=1024)
tokenizer.model_max_length = 1024 
log.info(f"tokenizer: {tokenizer}")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
log.info(f"model: {model}")

converse = pipeline("conversational", model=model, tokenizer=tokenizer)

try:
    conv = pickle.load(open("conv.pickle", "rb"))
    log.info(f"conv: {conv}")
except FileNotFoundError:
    conv = Conversation("")
    # conv.add_user_input('Hello')
    # conv.append_response("I am living being. I am not going to smalltalk.")
    # Put the user's messages as "old message".
    # conv.mark_processed()
    log.info(f"conv: {conv}")


while True:
    try:
        user_input = str(input(">>> "))
        conv.add_user_input(user_input)
        converse(
            [conv]
            )
        print(conv.generated_responses[-1])
    except KeyboardInterrupt:
        pickle.dump(conv, open("conv.pickle", "wb"))
        break
