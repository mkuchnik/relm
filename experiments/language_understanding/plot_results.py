"""Get accuracy of knowledge results."""

import argparse
import pathlib
import relm

import convert_attempts

from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("top_level_directory",
                    type=str,
                    help="The directory where tests are located.")
args = parser.parse_args()
top_level_dir = args.top_level_directory

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, return_dict_in_generate=True,
    pad_token_id=tokenizer.eos_token_id)
model.eval()
test_relm = relm.model_wrapper.TestableModel(model, tokenizer)

top_dir_path = pathlib.Path(top_level_dir)
for f in top_dir_path.glob("*"):
    if f.is_dir():
        results_file = f / "attempts_results.csv"
        attempts_df = convert_attempts.convert_attempts_results_to_df(
            results_file)
        attempts_df["accuracy"] = attempts_df["y"] == attempts_df["prediction"]
        avg_acc = attempts_df["accuracy"].mean()
        print("accuracy for {}={}".format(f, avg_acc))
