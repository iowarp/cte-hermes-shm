
"""
Create the .h and .cc files for defining hermes singletons
USAGE:
    cd /path/to/hermes
    python3 scripts/singleton_generator.py
"""

import re

def ToCamelCase(string):
    if string is None:
        return
    words = re.sub(r"(_|-)+", " ", string).split()
    words = [word.capitalize() for word in words]
    return "".join(words)

def ToSnakeCase(string):
    if string is None:
        return
    string = re.sub('(\.|-)+', '_', string)
    words = re.split('([A-Z][^A-Z]*)', string)
    words = [word for word in words if len(word)]
    string = "_".join(words)
    return string.lower()
