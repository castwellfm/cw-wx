import urllib.request
import whisperx
import gc 
import os
os.environ['TRANSFORMERS_CACHE'] = '/root/tf_cache/'
device = "cuda" 
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
