#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# working

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
import logging

log = logging.getLogger(__name__)
# add timestamp to log messages
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
log.info("Loaded dataset")
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
# ['<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>']

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
log.info("Transcription: " + transcription[0])