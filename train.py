# Fine-tuning ModernBERT-base to predict the grammaticality score   


# imports
from datasets import load_dataset                                     
from transformers import AutoTokenizer, set_seed, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from scipy.stats import spearmanr


# settings
SEED = 42
MODEL_NAME = "AI-Sweden-Models/ModernBERT-base"
MODEL_REVISION = "0bf738233da3c1bd32bc88cdbeb2c2c01dbac0eb"
DATA_FILE = "data/my_train.jsonl"
OUTPUT_DIR = "my_model"
# hyperparameters
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
WARMUP_RATIO = 0.1
MAX_LENGTH = 128


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    result = spearmanr(predictions, labels)
    return {"spearmanr": result.correlation}


def main():
    set_seed(SEED)

    data = load_dataset("json", data_files=DATA_FILE, split="train")
    data = data.train_test_split(test_size=0.1, seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    data = data.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH), batched=True)  
    data = data.rename_column("score", "labels")
    data = data.remove_columns(["token_type_ids"])                   

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, reference_compile=False, revision=MODEL_REVISION)

    args = TrainingArguments(
        output_dir="checkpoints",                                    
        save_strategy="no",                                           
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        logging_steps=50,
        seed=SEED,
        report_to="none",
        warmup_ratio=WARMUP_RATIO,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()                                                   

    print(trainer.evaluate())
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()