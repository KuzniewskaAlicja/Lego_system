from json import load

from .exceptions import *

with open('./config.json', 'r') as file:
    config = load(file)
