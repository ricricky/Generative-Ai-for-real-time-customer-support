import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

def prepare_dataset():
    # Load tokenizer
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define data paths
    data_files = {
        "train": "/home/ubuntu/sahil/email_response/daily_dialog_train.csv",
        "validation": "/home/ubuntu/sahil/email_response/daily_dialog_validation.csv",
        "test": "/home/ubuntu/sahil/email_response/daily_dialog_test.csv"
    }

    # Load dataset
    dataset = load_dataset("csv", data_files=data_files)



    def tokenize_function(examples):
        # Combine dialog and act for more context
        dialogues = [f"{dialog} {act} {emotion}" for dialog, act,emotion in zip(examples['dialog'], examples['act'],examples['emotion'])]

        # Tokenize the dialogues
        return tokenizer(dialogues, padding="max_length", truncation=True, max_length=128)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenizer, tokenized_datasets

def train_model():
    # Prepare dataset
    tokenizer, tokenized_datasets = prepare_dataset()

    # Load pre-trained model
    model_name = "microsoft/DialoGPT-medium"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare training arguments
    output_dir = "./smart_reply_model"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=400,
        save_steps=800,
        warmup_steps=500,
        prediction_loss_only=True,
    )

    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # For causal language modeling
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    train_model()
