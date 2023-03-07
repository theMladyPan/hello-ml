#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests

API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "The answer to the universe is [MASK].",
})

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output2 = query({
	"inputs": {
		"past_user_inputs": ["Which movie is the best ?"],
		"generated_responses": ["It's Die Hard for sure."],
		"text": "Can you explain why ?"
	},
})

API_URL = "https://api-inference.huggingface.co/models/microsoft/deberta-v3-large"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output3 = query({
	"inputs": "What is the answer to the universe ?",
})

print(
    output,
    output2,
    output3
)