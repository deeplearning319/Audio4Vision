import calendar
import time
import math
import re
import requests
from gtts import gTTS
from gtts_token.gtts_token import Token

def _patch_faulty_function(self):
    if self.token_key is not None:
        return self.token_key

    timestamp = calendar.timegm(time.gmtime())
    hours = int(math.floor(timestamp / 3600))

    results = requests.get("https://translate.google.com/")
    tkk_expr = re.search("(tkk:*?'\d{2,}.\d{3,}')", results.text).group(1)
    tkk = re.search("(\d{5,}.\d{6,})", tkk_expr).group(1)
    
    a , b = tkk.split('.')

    result = str(hours) + "." + str(int(a) + int(b))
    self.token_key = result
    return result


# Monkey patch faulty function.
Token._get_token_key = _patch_faulty_function

# Then call it normally.
tts = gTTS("Hello World")
tts.save('hello.mp3')