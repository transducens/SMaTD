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
) -> None:
    set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = LLMixtic(**model_params)

    # All train datasets are concatenated
    loaded_train_datasets, train_names = [], []
    for name, params in train_datasets.items():
        #dataset = load_from_disk(params["dataset_path"])[params["split"]]

        if params["dataset_path"].endswith(".jsonl"):
            dataset = load_dataset("json", data_files=params["dataset_path"])["train"]
        else:
            dataset = load_dataset("csv", data_files=params["dataset_path"], column_names=["src", "text", "label"], delimiter="\t", quoting=3)["train"]
            dataset = dataset.remove_columns("src")

        loaded_train_datasets.append(dataset)
        train_names.append(name)

    train_data = concatenate_datasets(loaded_train_datasets)
    train_name = "_and_".join(train_names)
    #train_data = train_data.select(range(10)) # OJO # TODO remove
    #train_data = concatenate_datasets([train_data] * 1000) # TODO remove

    # All dev datasets are concatenated
    loaded_dev_datasets, dev_names = [], []
    for name, params in dev_datasets.items():
        #dataset = load_from_disk(params["dataset_path"])[params["split"]]

        if params["dataset_path"].endswith(".jsonl"):
            dataset = load_dataset("json", data_files=params["dataset_path"])["train"]
        else:
            dataset = load_dataset("csv", data_files=params["dataset_path"], column_names=["src", "text", "label"], delimiter="\t", quoting=3)["train"]
            dataset = dataset.remove_columns("src")

        loaded_dev_datasets.append(dataset)
        dev_names.append(name)

    if len(loaded_dev_datasets) > 0:
        dev_data = concatenate_datasets(loaded_dev_datasets)
        dev_name = "_and_".join(dev_names)
        #dev_data = dev_data.select(range(10)) # OJO # TODO remove
    else:
        dev_data = None
        dev_name = None

    # Fit model
    model.fit(train_data, train_name, eval_dataset=dev_data, cache_name_eval=dev_name)

    if len(loaded_dev_datasets) > 0:
        # Dev results
        is_training = model.model.training

        model.model.eval()

        with torch.no_grad():
            dev_preds = model.predict(dev_data, dev_name)

        if is_training:
            model.model.train()

        dev_results = evaluate_classification(dev_data["label"], dev_preds, model)

        print(f"Dev results: {dev_results}")

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

    print(f"Test results: {all_results}")


def run_importance(
    train_datasets: Dict[str, Dict[str, str]],
    test_datasets: Dict[str, Dict[str, str]],
    model_params: Dict,
    save_dir: Path,
    dev_datasets: Dict[str, Dict[str, str]] = {},
    seed=0,
) -> None:
    assert dev_datasets == {}, "Dev datasets not supported for importance"
    set_seed(seed)
    if save_dir.exists():
        save_dir = save_dir / f"{save_dir.name}_importance"

    save_dir.mkdir(parents=True, exist_ok=True)
    model = LLMixtic(**model_params)

    assert (
        len(test_datasets) == 1
    ), "It is recommended to only run importance on one dataset"

    name, params = next(iter(test_datasets.items()))
    dataset = load_from_disk(params["dataset_path"])[params["split"]]
    importances = model.permutation_feature_importance(dataset, name)

    for name, df in importances.items():
        df.to_csv(save_dir / f"{name}.tsv", sep="\t", index=False)
        df.to_markdown(save_dir / f"{name}.md", index=False)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--importance", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print(f"Args: {args}")

    config = parse_config(args.config)
    save_dir = args.save_dir / f"{config['run_name']}_seed_{args.seed}"

    for arg in ("training_params", "inference_params"):
        if arg in config["model"] and "output_dir" in config["model"][arg]:
            config["model"][arg]["output_dir"] = config["model"][arg]["output_dir"].rstrip("/") + f"_seed_{args.seed}"

    assert (
        args.train != args.importance
    ), "Only one of --train and --importance can be chosen"
    run = run_single if args.train else run_importance

    run(config["train"], config["test"], config["model"], save_dir,
        dev_datasets=config.get("dev", {}), seed=args.seed)
