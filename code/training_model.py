import sys
import os
import tiktoken
import torch
import time
from torch.utils.data import Dataset , DataLoader
from create_dataset_class import Spam_Dataset
import matplotlib.pyplot as plt

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

def calc_accuracy_loader(data_loader, model,device, num_batches = None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)

    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i< num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, : ]
            predicted_labels = torch.argmax(logits, dim =-1)

            num_examples += predicted_labels.shape[0]

            correct_predictions +=(
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions/num_examples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)



tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = Spam_Dataset(csv_file = "../code/sms_spam_collection/train.csv", tokenizer = tokenizer, max_length = None)
val_dataset = Spam_Dataset(csv_file = "../code/sms_spam_collection/val.csv", tokenizer = tokenizer, max_length = None)
test_dataset = Spam_Dataset(csv_file = "../code/sms_spam_collection/test.csv", tokenizer = tokenizer, max_length = None)


num_workers =0
batch_size =8
torch.manual_seed(123)
train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = num_workers,
                          drop_last = True)

val_loader = DataLoader(val_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = num_workers,
                          drop_last = True)

test_loader = DataLoader(test_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = num_workers,
                          drop_last = True)



train_accuracy = calc_accuracy_loader(train_loader,
                                      model, device, num_batches =10)
val_accuracy = calc_accuracy_loader(val_loader,
                                      model, device, num_batches =10)
test_accuracy = calc_accuracy_loader(test_loader,
                                      model, device, num_batches =10)

# print(f"Training accuracy : {train_accuracy*100: .2f}%")
# print(f"Validation accuracy : {val_accuracy*100: .2f}%")
# print(f"Test accuracy : {test_accuracy*100: .2f}%")


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    
    elif num_batches is None:
        num_batches =  len(data_loader)

    else:
        num_batces = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batces:
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            total_loss += loss
        else:
            break
    return total_loss/num_batches

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches = 5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches = 5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches = 5)

# print(f"Training loss : {train_loss: .3f}%")
# print(f"Validation loss : {val_loss: .3f}%")
# print(f"Test loss : {test_loss: .3f}%")


def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    
    train_losses , val_losses, train_acc, val_acc = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen+= input_batch.shape[0]
            global_step+=1

            if global_step % eval_freq ==0:
                train_loss , val_loss = evaluate_model(model, train_loader, val_loader, device, 
                                                       eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Epoch {epoch+1} (step {global_step: 06d}): "
                      f"train loss {train_loss :.3f}, "
                      f"Val loss {val_loss} : .3f")
                
            train_accuracy = calc_accuracy_loader(
                train_loader, model, device, num_batches = eval_iter)
            val_accuracy = calc_accuracy_loader(
                val_loader, model, device, num_batches = eval_iter)
            
            print(f"Training accuracy : {train_accuracy*100 : .2f}% | ", end = "")
            print(f"Validation accuracy : {val_accuracy*100 : .2f}% ")
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)

    return train_losses, val_losses, train_acc, val_acc, examples_seen
            
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches = eval_iter
        )

        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()
    return train_loss, val_loss
def main():

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, weight_decay = 0.1)

    num_epochs =5

    train_losses, val_losses, train_acc, val_acc , examples_seen = train_classifier_simple(model,
                                                                                        train_loader, val_loader, optimizer, device,
                                                                                        num_epochs = num_epochs, eval_freq = 50,
                                                                                        eval_iter = 5)

    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Training completed in {execution_time:.2f} minutes ")



    def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label="loss"):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        #1
        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()
        #2
        ax2 = ax1.twiny()
        ax2.plot(examples_seen, train_values, alpha=0) #3
        ax2.set_xlabel("Examples seen")
        fig.tight_layout() #4
        plt.savefig(f"{label}-plot.pdf")
        plt.show()
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_acc))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_acc))
    plot_values(
    epochs_tensor, examples_seen_tensor, train_acc, val_acc,
    label="accuracy"
    )

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    def classify_review(text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):
        model.eval()
        input_ids = tokenizer.encode(text) #1
        supported_context_length = model.pos_emb.weight.shape[1]
        input_ids = input_ids[:min( #2
        max_length, supported_context_length
        )]
        input_ids += [pad_token_id] * (max_length - len(input_ids)) #3
        input_tensor = torch.tensor(
        input_ids, device=device
        ).unsqueeze(0) #4
        with torch.no_grad(): #5
            logits = model(input_tensor)[:, -1, :] #6
        predicted_label = torch.argmax(logits, dim=-1).item()
        return "spam" if predicted_label == 1 else "not spam"
    
    text_1 = (
"You are a winner you have been specially"
" selected to receive $1000 cash or a $2000 award."
)
    print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))


    torch.save(model.state_dict(), "review_classifier.pth")

if __name__ == '__main__':
    main()