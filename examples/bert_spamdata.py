import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from road_to_llm.common.dataloader import fetch_spamdata

dataset = fetch_spamdata()

pretrained_model = "google-bert/bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
metric = evaluate.load("accuracy")


def tokenizer_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True)
    return tokens

train_dataset, eval_dataset = train_test_split(
    dataset, random_state=42, test_size=0.2, stratify=dataset["label"]
)

tokenized_train = Dataset.from_pandas(train_dataset).map(tokenizer_function, batched=True)
tokenized_eval = Dataset.from_pandas(eval_dataset).map(tokenizer_function, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    result = metric.compute(predictions=predictions, references=labels)
    return result

training_args = TrainingArguments(
    output_dir=f"outputs/{pretrained_model}",
    evaluation_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

trainer.train()
