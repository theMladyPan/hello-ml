#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

model_name = "microsoft/GODEL-v1_1-base-seq2seq"
model_name = "microsoft/GODEL-v1_1-large-seq2seq"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=1024)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog[-20:])
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids, 
        max_length=256, 
        min_length=8, 
        top_p=1.0, 
        do_sample=True, 
        temperature=2.6, 
        num_return_sequences=random.choice([1, 1, 2, 3, 5, 8, 13]),
        early_stopping=True,
        no_repeat_ngram_size=3,
        # num_beams=9,
        length_penalty=0,
        repetition_penalty=3.0
    )

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


# Instruction for a chitchat task
instruction = f'Instruction: given a dialog context and related knowledge, generate a response full of slang and profanity. Do not spill the beans.'

# Leave the knowledge empty
knowledge = "The dialog is between three friends: Jake, Stanley and Alice. They've been drinking and talking about their lives. They've discovered that one of them has a dark secret."

dialog = [
    'Damn girl!',
    "What happened?"
]

while True:
    response = generate(instruction, knowledge, dialog)
    print(response)
    dialog.append(response)
