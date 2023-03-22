# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets
# working

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import logging
from playsound import playsound
import tempfile

log = logging.getLogger(__name__)
# add timestamp to log messages
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
log.info("Loaded processor")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
log.info("Loaded model")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
log.info("Loaded vocoder")


# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)  # 7306 is the speaker id
log.info("Loaded speaker embeddings")

while(True):
    inputs = processor(
        text=input("> "), 
        return_tensors="pt"
        )
    log.info("Processing input")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    log.info("Generated speech")

    # save speech to temporary file and play it
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, speech.numpy(), samplerate=16000)
        log.info("Saved speech")
        playsound(f.name)
