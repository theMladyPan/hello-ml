#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random
from colors import red, green, yellow, blue, magenta, cyan, white

model_name = "microsoft/GODEL-v1_1-large-seq2seq"
model_name = "microsoft/GODEL-v1_1-base-seq2seq"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=1024)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids, 
        max_length=128, 
        min_length=8, 
        top_p=.95, 
        do_sample=True, 
        temperature=1.1, 
        num_return_sequences=random.choice([1, 1, 2, 3, 5]),
        early_stopping=True,
        no_repeat_ngram_size=3,
        # num_beams=9,
        length_penalty=-1.0,
        repetition_penalty=3.0
    )

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


# Instruction for a chitchat task
instruction = f"Instruction: given a dialog context, continue in dialog between two friends. Generate possible responses and defent eachother's responses."

# Leave the knowledge empty
knowledge = "This is conversation between two friends."

dialog = [
    "Why are my neutral grip pull-ups way easier than normal grip?"
    "Should I train with neutral grip because it allows for more volume this way?"
    "I ask because I do 50 pull-ups at the start of my upper body workout, but 50 neutral "
    "grip pull-ups is doable in decent time and sets. Normal pull-ups are so fucking hard and "
    "I barely get to do them all, it doesn't even feel like it's the same fucking exercise",
    "Don't quote me on that but I think with neutral grip your lats are more active, thus making it easier. "
    "Atleast I've heard that neutral grip lat pulldowns train lats more so that's my guess.",
    "When I train chinups my pullup numbers stagnate, when I train pullups both my pullup and chinup numbers go up. "
    "So anecdotally I believe pullups are a better exercise overall in most cases.",
    "close. neutral grip lets you cheat and use your biceps for a neutral grip chin-up instead of a pull up.",
    "In my experience neutral grip pullups are also easier than chinups. I think it's cuz you can actually "
    "squeeze out a decent amount of chest activation on neutral grip but I'm not sure."
]

while True:
    try:
        response1 = generate(instruction, knowledge, dialog)
        print(red(response1))
        dialog.append(response1)
        
        response2 = input("> ")
        dialog.append(response2)
    except KeyboardInterrupt:
        break
