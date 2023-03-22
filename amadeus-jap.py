#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# not working - dependency issues

from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
from playsound import playsound
import logging
log = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)
import tempfile

model = Text2Speech.from_pretrained("mio/amadeus")

speech, *_ = model(u"おはようございます。")

with tempfile.NamedTemporaryFile(suffix=".wav") as f:
    sf.write(f.name, speech, samplerate=16000)
    log.info("Saved speech")
    playsound(f.name)