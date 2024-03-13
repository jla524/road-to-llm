import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

dataset = load_dataset("squad")

model_name = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
special_tokens = tokenizer.special_tokens_map


def add_end_token_to_question(input_dict):
    input_dict["question"] += special_tokens["bos_token"]
    return input_dict


def tokenize_function(input_dict):
    return tokenizer(input_dict["question"], truncation=True)


def divide_tokenized_text(tokenized_text_dict, block_size=128):
    concatenated_examples = {k: sum(tokenized_text_dict[k], []) for k in tokenized_text_dict.keys()}
    total_length = len(concatenated_examples[list(tokenized_text_dict.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

dataset = dataset.remove_columns(["id", "title", "context", "answers"])
dataset = dataset.map(add_end_token_to_question)
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["question"])
dataset = dataset.map(divide_tokenized_text, batched=True, batch_size=1000)

training_args = TrainingArguments(f"outputs/{model_name}", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)
trainer.train()

results = trainer.evaluate()
print(f"Perplexity: {math.exp(results['eval_loss']):.2f}")
