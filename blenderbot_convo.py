#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from pprint import pprint as print

API_TOKEN = "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"

API_URL0 = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"
API_URL1 = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-1B-distill"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

past_user_inputs = []
generated_responses = []
	
while True:
    new_input = str(input(">>> "))
    output = query({
        "inputs": {
            "past_user_inputs": past_user_inputs[:],
            "generated_responses": generated_responses[:],
            "text": new_input
        }
    })
    
    new_output = output.get("generated_text")
    if new_output is None:
        print(output)
        
    else:
        print(new_output)    
        past_user_inputs.append(new_input)
        generated_responses.append(new_output)
    