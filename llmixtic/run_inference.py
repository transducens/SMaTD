import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd
import torch
from _jsonnet import evaluate_file
from datasets import concatenate_datasets, load_from_disk, load_dataset
from sklearn.metrics import accuracy_score, f1_score

from src.llmixtic import LLMixtic


def set_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_classification(
    refs: List[int], preds: List[int], model: LLMixtic
) -> Dict:
    return {
        "features": ", ".join(model.feature_params["features"]),
        "models": ", ".join(model.feature_params["models"]),
        "accuracy": accuracy_score(refs, preds),
        "macro-f1": f1_score(refs, preds, average="macro"),
        "micro-f1": f1_score(refs, preds, average="micro"),
    }


def parse_config(
    path: Path,
) -> Dict:
    config = json.loads(evaluate_file(str(path)))
    assert all(x in config for x in {"run_name", "train", "test", "model"})

    return config


def run_single(
    train_datasets: Dict[str, Dict[str, str]],
    test_datasets: Dict[str, Dict[str, str]],
    model_params: Dict,
    save_dir: Path,
    dev_datasets: Dict[str, Dict[str, str]] = {},
    seed=0,
    only_scores=False,
) -> None:
    set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = LLMixtic(**model_params)

    # All train datasets are concatenated
    loaded_train_datasets, train_names = [], []
    #for name, params in train_datasets.items():
    #    #dataset = load_from_disk(params["dataset_path"])[params["split"]]
    #    dataset = load_dataset("csv", data_files=params["dataset_path"], column_names=["src", "text", "label"], delimiter="\t", quoting=3)["train"]
    #    dataset = dataset.remove_columns("src")
    #    loaded_train_datasets.append(dataset)
    #    train_names.append(name)

    #train_data = concatenate_datasets(loaded_train_datasets)
    #train_name = "_and_".join(train_names)
    #train_data = train_data.select(range(10)) # OJO # TODO remove
    #train_data = concatenate_datasets([train_data] * 1000) # TODO remove

    # All dev datasets are concatenated
    #loaded_dev_datasets, dev_names = [], []
    #for name, params in dev_datasets.items():
    #    #dataset = load_from_disk(params["dataset_path"])[params["split"]]
    #    dataset = load_dataset("csv", data_files=params["dataset_path"], column_names=["src", "text", "label"], delimiter="\t", quoting=3)["train"]
    #    dataset = dataset.remove_columns("src")
    #    loaded_dev_datasets.append(dataset)
    #    dev_names.append(name)

    #if len(loaded_dev_datasets) > 0:
    #    dev_data = concatenate_datasets(loaded_dev_datasets)
    #    dev_name = "_and_".join(dev_names)
    #    #dev_data = dev_data.select(range(10)) # OJO # TODO remove
    #else:
    #    dev_data = None
    #    dev_name = None

    # Fit model
    #model.fit(train_data, train_name, eval_dataset=dev_data, cache_name_eval=dev_name)

    #if len(loaded_dev_datasets) > 0:
    #    # Dev results
    #    is_training = model.model.training
#
    #    model.model.eval()
#
    #    with torch.no_grad():
    #        dev_preds = model.predict(dev_data, dev_name)
#
    #    if is_training:
    #        model.model.train()
#
    #    dev_results = evaluate_classification(dev_data["label"], dev_preds, model)
#
    #    print(f"Dev results: {dev_results}")

    # The test datasets are considered one-by-one
    all_results = {}
    for name, params in test_datasets.items():
        #dataset = load_from_disk(params["dataset_path"])[params["split"]]

        if params["dataset_path"].endswith(".jsonl"):
            dataset = load_dataset("json", data_files=params["dataset_path"])["train"]
        else:
            dataset = load_dataset("csv", data_files=params["dataset_path"], column_names=["src", "text", "label"], delimiter="\t", quoting=3)["train"]
            dataset = dataset.remove_columns("src")

        #dataset = dataset.select(range(10)) # OJO # TODO remove
        preds = model.predict(dataset, name)

        if len(preds.shape) > 1:
            assert preds.shape[-1] == 1, f"Expected 1D tensor, got {preds.shape}"
            assert len(preds.shape) == 2

            preds2 = preds[:, 0] # equivalent to preds.squeeze()
        else:
            preds2 = preds

        if only_scores:
            assert len(dataset["label"]) == len(preds2), f"Length mismatch between refs and preds: {len(dataset['label'])} vs {len(preds2)}"

            for i in range(len(preds2)):
                #print('1' if preds2[i].item() == dataset["label"][i] else '0') # accuracy per instance
                print(preds2[i].item()) # inference output per instance

        results = evaluate_classification(dataset["label"], preds, model)
        all_results[name] = results

        df = pd.DataFrame({name: results}).T.reset_index()
        df.columns = ["dataset"] + list(results.keys())
        df.to_csv(save_dir / f"{name}.tsv", sep="\t", index=False)
        df.to_markdown(save_dir / f"{name}.md", index=False)

    df = pd.DataFrame(all_results).T.reset_index()
    df.columns = ["dataset"] + list(results.keys())
    df.to_csv(save_dir / "all_test.tsv", sep="\t", index=False)
    df.to_markdown(save_dir / "all_test.md", index=False)

    if not only_scores:
        print(f"Test results: {all_results}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained-model-path", type=str)
    parser.add_argument("--only-scores", action="store_true")

    args = parser.parse_args()

    print(f"Args: {args}")

    config = parse_config(args.config)
    save_dir = args.save_dir / f"{config['run_name']}_seed_{args.seed}"

    for arg in ("training_params", "inference_params",):
        if arg in config["model"] and config["model"][arg] is not None and "output_dir" in config["model"][arg] and isinstance(config["model"][arg]["output_dir"], str):
            config["model"][arg]["output_dir"] = config["model"][arg]["output_dir"].rstrip("/") + f"_seed_{args.seed}"

    if args.pretrained_model_path:
        print(f"args.pretrained_model_path: {args.pretrained_model_path}")

        given_model = config["model"]["model_params"].get("pretrained_model_name_or_path", None)

        if given_model is not None:
            print(f"Overriding given pretrained model path {given_model} with {args.pretrained_model_path}")

        config["model"]["model_params"]["pretrained_model_name_or_path"] = args.pretrained_model_path

    assert config["model"]["model_params"].get("pretrained_model_name_or_path", None) is not None, "A pretrained model path must be provided for inference"

    run_single(config["train"], config["test"], config["model"], save_dir,
               dev_datasets=config.get("dev", {}), seed=args.seed, only_scores=args.only_scores)
