# make_data.py
# for building the training data: toy data + DaLAJ + ScaLA -> data/my_train.jsonl

# imports
import re
import pandas as pd
from datasets import load_dataset


# settings 
SEED = 42
GOOD_SCORE = 0.95
BAD_SCORE = 0.4
OUTPUT_FILE = "data/my_train.jsonl"
DALAJ_REVISION = "ad9e666c82b639eec7e2e8733178cbb1e2f3e6ef"
SCALA_REVISION = "aded78beae37445bf3917102d0b332049cfc8c99"


def fix_spacing(text):
    return re.sub(r"\s+([.,!?:;])", r"\1", text)


def load_toy():
    df = pd.read_json("data/train.jsonl", lines=True)
    return df[["text", "score"]]


def load_dalaj():
    dalaj = load_dataset("mteb/DalajClassification", split="train", revision=DALAJ_REVISION)
    dalaj_df = dalaj.to_pandas()
    dalaj_df["text"] = dalaj_df["text"].apply(fix_spacing)
    dalaj_df["score"] = dalaj_df["label"].apply(lambda x: GOOD_SCORE if x == 0 else BAD_SCORE)
    return dalaj_df[["text", "score"]]


def load_scala():
    scala = load_dataset("mteb/scala_sv_classification", split="full_train", revision=SCALA_REVISION)
    scala_df = scala.to_pandas()
    scala_df["score"] = scala_df["label"].apply(lambda x: GOOD_SCORE if x == "correct" else BAD_SCORE)
    return scala_df[["text", "score"]]


def main():
    toy_df = load_toy()
    dalaj_df = load_dalaj()
    scala_df = load_scala()
    train_df = pd.concat([toy_df, dalaj_df, scala_df], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED)
    train_df.to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)
    print("Number of rows:", len(train_df))
    print(train_df["score"].round(1).value_counts())
    

if __name__ == "__main__":
     main()