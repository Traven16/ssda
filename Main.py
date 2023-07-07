import torch
from torch.utils.data import Dataset
from PIL import Image

class SSDADataset(Dataset):
    def __init__(self, root_dir, pics, lines, processor, max_target_length=128, mode="train"):
        self.root_dir = root_dir
        self.mode = mode
        self.pics = pics
        self.lines = lines
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.pics[idx]
        text = self.lines[idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        if self.mode != "test":
    
            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        else:
            encoding = {"pixel_values": pixel_values.squeeze()}
        return encoding


filepath = "../../../../../data/traven/ssda/lines/transcriptions.csv"
base_dir = "../../../../../data/traven/ssda/lines/data/"

filepath_eval = "../../../../../data/traven/ssda/data/validation/gt.tsv"
base_dir_eval = "../../../../../data/traven/ssda/data/validation/"

filepath_test = "../../../../../data/traven/ssda/test/files.tsv"
base_dir_test = "../../../../../data/traven/ssda/test/"

def read(path):
    pics, lines = [], []
    with open(path, "r") as f:
        data = f.readlines()
        for n in data:
            p, l = n.split("\t")
            pics.append(p)
            lines.append(l)
    return pics, lines


pics_train, lines_train = read(filepath)
pics_val, lines_val = read(filepath_eval)
pics_test, lines_test = read(filepath_test)
#pics_test = pics_test[:10]
#lines_test = lines_test[:10]
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
train_dataset = SSDADataset(base_dir, pics_train, lines_train, processor)
eval_dataset = SSDADataset(base_dir_eval, pics_val, lines_val, processor)
test_dataset = SSDADataset(base_dir_test, pics_test, lines_test, processor, mode="test")

print(len(train_dataset))
encoding = train_dataset[0]
for k,v in encoding.items():
    print(k, v.shape)


# Load model
from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# Config

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True, 
    output_dir="../../../../../../../data/traven/ssda_model_LARGE/",
    logging_steps=200,
    save_steps=2000,
    eval_steps=5600,
    num_train_epochs=4,
)

     

     
from datasets import load_metric

cer_metric = load_metric("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}



from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

import csv
def save_lists_as_tsv(list1, list2, tsv_file):
    with open(tsv_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for item1, item2 in zip(list1, list2):
            writer.writerow([item1, item2])



if False:
    trainer.train()
else:
    model = VisionEncoderDecoderModel.from_pretrained(
    f"../../../../../../../data/traven/ssda_model_LARGE/checkpoint-10000")
    out = trainer.predict(test_dataset)
    text = processor.batch_decode(out[0], skip_special_tokens=True)

    save_lists_as_tsv(pics_test, text, f"../../../../../../../data/traven/ssda_out_1.tsv")


