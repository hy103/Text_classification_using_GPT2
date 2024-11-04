import torch
from training_model import calc_accuracy_loader
from create_dataset_class import Spam_Dataset
import tiktoken
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

# Load the checkpoint
checkpoint_path = "trained_gpt_model.pth"
checkpoint = torch.load(checkpoint_path)

# Load the saved config and model architecture (use the config saved in the checkpoint)
model_config = checkpoint["model_config"]

# Create a model matching the saved architecture and load state dict
model = torch.nn.Sequential(
    torch.nn.Linear(model_config["emb_dim"], 2, bias=False)  # Example output layer; customize as needed
)
model.load_state_dict(checkpoint["model_state_dict"])

# Initialize optimizer and load optimizer state
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Load training stats from checkpoint
train_losses, val_losses = checkpoint["train_losses"], checkpoint["val_losses"]
train_acc, val_acc = checkpoint["train_acc"], checkpoint["val_acc"]
examples_seen = checkpoint["examples_seen"]

# Setup tokenizer and datasets
tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = Spam_Dataset(csv_file="../code/sms_spam_collection/train.csv", tokenizer=tokenizer, max_length=None)
val_dataset = Spam_Dataset(csv_file="../code/sms_spam_collection/val.csv", tokenizer=tokenizer, max_length=None)
test_dataset = Spam_Dataset(csv_file="../code/sms_spam_collection/test.csv", tokenizer=tokenizer, max_length=None)

num_workers = 0
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

# Plotting
num_epochs = 5
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

epochs_tensor = torch.linspace(0, num_epochs, len(train_acc))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_acc))
plot_values(epochs_tensor, examples_seen_tensor, train_acc, val_acc, label="accuracy")

# Calculate final accuracies
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
