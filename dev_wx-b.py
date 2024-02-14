#import urllib.request
import whisperx
#import gc 
#import os
#os.environ['TRANSFORMERS_CACHE'] = '/root/tf_cache/'
#device = "cuda" 
#batch_size = 16 # reduce if low on GPU mem
#compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
#model = whisperx.load_model("large-v2", device, compute_type=compute_type)
#model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

import faster_whisper
import os
os.environ['TRANSFORMERS_CACHE'] = '/root/tf_cache/'
from faster_whisper.utils import download_model, format_timestamp, get_logger
download_model("large-v2")
whisperx.load_align_model(language_code="en", device="cpu")

import torchaudio
bundle = torchaudio.pipelines.MMS_FA
t_model = bundle.get_model(with_star=False)
LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)
