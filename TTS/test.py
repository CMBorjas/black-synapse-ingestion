import pygame._sdl2.audio as sdl2_audio
from pygame import mixer
mixer.init()
print(sdl2_audio.get_audio_device_names(False))
