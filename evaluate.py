from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("distilbert-imdb/")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-imdb/")

from datasets import load_dataset
imdb = load_dataset("imdb")
small_test_dataset = imdb["test"].shuffle(seed=42) #.select([i for i in list(range(300))])

def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)
 
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import numpy as np
from datasets import load_metric

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

from transformers import TrainingArguments, Trainer

trainer = Trainer(
   model=model,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)

results = trainer.evaluate()
print(results)


