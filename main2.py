from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import cv2

import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

# Fancy workaround
import calendar
import time
import math
import re
import requests
from gtts import gTTS
from gtts_token.gtts_token import Token

from picamera import PiCamera
from time import sleep

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "./model.ckpt-2000000",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "./word_counts.txt", "Text file containing the vocabulary.")
#image_name = input("Enter name of image (e.g. 2.jpg):")

tf.logging.set_verbosity(tf.logging.INFO)

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

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  camera=PiCamera()
  camera.rotation = 180
  camera.resolution = (1280, 720)
  camera.framerate = 40
  
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    generator = caption_generator.CaptionGenerator(model, vocab)
    iter = 0
    
    while(iter<25):
      i = 0
      while(os.path.isfile('files/' + str(i) + '.txt')):
        i += 1
      filename = 'files/' + str(i) + '.jpg'
      
      print('Capturing image')
      camera.start_preview(alpha=250)
      sleep(10)
      camera.capture('files/' + str(i) + '.jpg')
      camera.stop_preview()
      print('Captured and saved image')
      print('Generating caption')
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Caption for image %s:" % os.path.basename(filename))
      top_caption = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
      top_caption = " ".join(top_caption)
      print(top_caption)
      print("Creating speech")
      # Monkey patch faulty function.
      Token._get_token_key = _patch_faulty_function
      audio = gTTS(text=top_caption,lang='en')
      with open('files/' + str(i) + '.txt', 'w') as f:
        f.write(top_caption)
      print("Saving speech")
      audio.save('files/' + str(i) + '.mp3')
      print("Playing speech")
      os.system('mpg321 ' + 'files/' + str(i) + '.mp3')
      img=cv2.imread('files/' + str(i) + '.jpg')
      print("Showing image with caption")
      cv2.putText(img,top_caption, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
      img1=cv2.resize(img,(640,360),interpolation = cv2.INTER_AREA)
      cv2.imwrite('files/' + str(i) + '_cap'+'.jpg',img1)
      img2=cv2.imread('files/' + str(i) + '_cap'+'.jpg')
      cv2.imshow('image',img2)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      iter += 1
            

if __name__ == "__main__":
  tf.app.run()
