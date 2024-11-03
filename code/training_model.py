import sys
import os
import tiktoken
import torch

# Add the path to Language_model/code to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Language_model', 'code')))

from GPT_Model import GPTModel
from Loading_pretrained_weights import load_weights_into_gpt
from text_token_text import text_to_token_ids, token_ids_to_text, generate_text_simple


CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Arguments are extremely vulgar"
BASE_CONFIG = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "drop_rate" : 0.0,
    "qkv_bias" : True
}

model_configs = {
    "gpt2-small (124M)" : {"emb_dim": 768, "n_layers" : 12, "n_heads" : 12},
    "gpt2-small (355M)" : {"emb_dim": 1024, "n_layers" : 24, "n_heads" : 16},
    "gpt2-small (774M)" : {"emb_dim": 1280, "n_layers" : 36, "n_heads" : 20},
    "gpt2-small (1558M)" : {"emb_dim": 1600, "n_layers" : 48, "n_heads" : 25}
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

tokenizer = tiktoken.get_encoding("gpt2")


model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Language_model', 'code', 'model_settings_params.pth')
print(model_path)
model = torch.load(model_path)
setttings, params = model["model_settings_dict"], model["model_params_dict"]

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

text_1 = "Arguments are extremely vulgar"
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(text_1, tokenizer),
    max_new_tokens= 15,
    context_size=CHOOSE_MODEL["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))