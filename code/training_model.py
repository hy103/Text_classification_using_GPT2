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

model_path = "/Users/harshayarravarapu/Documents/GitHub/Language_model/code/model_settings_params.pth"
model = torch.load(model_path)
setttings, params = model["model_settings_dict"], model["model_params_dict"]

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# text_1 = "Arguments are extremely vulgar"
# token_ids = generate_text_simple(
#     model = model,
#     idx = text_to_token_ids(text_1, tokenizer),
#     max_new_tokens= 15,
#     context_size=BASE_CONFIG["context_length"]
# )

# print(token_ids_to_text(token_ids, tokenizer))


# text_2 = (
#     "Is the following text 'spam'? Answer with 'yes' or 'no : "
#     " 'You are a winner you have been specially"
#     " selected to receive $1000 cash or a $2000 award. ' "
# )

# token_ids = generate_text_simple(
#     model = model, 
#     idx = text_to_token_ids(text_2, tokenizer),
#     max_new_tokens = 23,
#     context_size = BASE_CONFIG["context_length"]
# )

# print(token_ids_to_text(token_ids, tokenizer))

#*******************************************************************#
#***************    GPT Model architecture        ******************#
#*******************************************************************#
# GPTModel(
#   (tok_emb): Embedding(50257, 768)
#   (pos_emb): Embedding(1024, 768)
#   (drop_emb): Dropout(p=0.0, inplace=False)
#   (trf_blocks): Sequential(
#     (0): Short_transformerblock(
#       (att): Multihead_attention(
#         (W_query): Linear(in_features=768, out_features=768, bias=True)
#         (W_keys): Linear(in_features=768, out_features=768, bias=True)
#         (W_values): Linear(in_features=768, out_features=768, bias=True)
#         (out_proj): Linear(in_features=768, out_features=768, bias=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#       )
#       (ff): FeedForward_network(
#         (layers): Sequential(
#           (0): Linear(in_features=768, out_features=3072, bias=True)
#           (1): GELU()
#           (2): Linear(in_features=3072, out_features=768, bias=True)
#         )
#       )
#       (layernorm1): LayerNorm()
#       (layernorm2): LayerNorm()
#       (drop_short): Dropout(p=0.0, inplace=False)
#     )
#     (1): Short_transformerblock(
#       (att): Multihead_attention(
#         (W_query): Linear(in_features=768, out_features=768, bias=True)
#         (W_keys): Linear(in_features=768, out_features=768, bias=True)
#         (W_values): Linear(in_features=768, out_features=768, bias=True)
#         (out_proj): Linear(in_features=768, out_features=768, bias=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#       )
#       (ff): FeedForward_network(
#         (layers): Sequential(
#           (0): Linear(in_features=768, out_features=3072, bias=True)
#           (1): GELU()
#           (2): Linear(in_features=3072, out_features=768, bias=True)
#         )
#       )
#       (layernorm1): LayerNorm()
#       (layernorm2): LayerNorm()
#       (drop_short): Dropout(p=0.0, inplace=False)
#     )
    #### Removed layers from 2-10 as they are similar and reducing the space to look better
#     (11): Short_transformerblock(
#       (att): Multihead_attention(
#         (W_query): Linear(in_features=768, out_features=768, bias=True)
#         (W_keys): Linear(in_features=768, out_features=768, bias=True)
#         (W_values): Linear(in_features=768, out_features=768, bias=True)
#         (out_proj): Linear(in_features=768, out_features=768, bias=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#       )
#       (ff): FeedForward_network(
#         (layers): Sequential(
#           (0): Linear(in_features=768, out_features=3072, bias=True)
#           (1): GELU()
#           (2): Linear(in_features=3072, out_features=768, bias=True)
#         )
#       )
#       (layernorm1): LayerNorm()
#       (layernorm2): LayerNorm()
#       (drop_short): Dropout(p=0.0, inplace=False)
#     )
#   )
#   (final_norm): LayerNorm()
#   (out_head): Linear(in_features=768, out_features=50257, bias=False)
# )


#### FREEZING THE MODEL #####
for param in model.parameters():
    param.require_grad = False

### Training the last layer of the model,
### by defaualt the last layer require_grad is set to True 
###    so no need to put require_grad= true
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features = BASE_CONFIG["emb_dim"], ## the emb dim is 768 same as the GPT model emb_dim 
                                 out_features = num_classes, bias = False)


### For better improvement in performance
### We can also train the layer of final norm
### We can also trian the last transformer block

for param in model.trf_blocks[-1].parameters():
    param.require_grad = True
for param in model.final_norm.parameters():
    param.require_grad = True

def calc_accuracy_loader(data_loader, model, num_batches = None):
    model.eval()
    correct_predictions, num_batches = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)

    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i< num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[: -1 : ]
            predicted_labels = torch.argmax(logits, dim =-1)

            num_examples += predicted_labels.shape[0]

            current_predictions +=(
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break


    return correct_predictions/num_examples


