#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip install -i https://pypi.fury.io/balacoon/ balacoon-tts
# not working, dependency issues

import wave
from balacoon_tts import TTS
# adjust the path to the addon based on the previous step
tts = TTS("./en_us_cmuartic_jets_cpu.addon")
# this will return a list of speakers that model supports.
supported_speakers = tts.get_speakers()
speaker = supported_speakers[-1]
# finally run synthesis
samples = tts.synthesize("hello world", speaker)
# up to you what to do with the synthesized samples (np.int16 array)
# in this example we will save them to a file
with wave.open("example.wav", "w") as fp:
    fp.setparams((1, 2, 24000, len(samples), "NONE", "NONE"))
    fp.writeframes(samples)