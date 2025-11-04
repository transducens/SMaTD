local language = "de-en";
local mt = "Unbabel_TowerInstruct-7B-v0.2";
local run_name = std.format("mtdetect_%s_%s", [mt, language]);

local train_data = {
    "mtdetect_train": {
        dataset_path: std.format("/home/cgarcia/Documentos/mtdetect/mtdetect/wmt_data/%s-mtpaper.all.sentences.shuf.%s.%s.out.all.shuf", [language, "train", mt]),
        split: "train",
    },
};

local dev_data = {
    "mtdetect_dev": {
        dataset_path: std.format("/home/cgarcia/Documentos/mtdetect/mtdetect/wmt_data/%s-mtpaper.all.sentences.shuf.%s.%s.out.all.shuf", [language, "train", mt]),
        split: "dev",
    },
};

local test_data = {
    "mtdetect_test": {
        dataset_path: std.format("/home/cgarcia/Documentos/mtdetect/mtdetect/wmt_data/%s-mtpaper.all.sentences.shuf.%s.%s.out.all.shuf", [language, "train", mt]),
        split: "test",
    },
};

local model = {
    "feature_params": {
        "models": [
            #"gpt2",
            #"gpt2-medium",
            #"distilgpt2",
            #"meta-llama/LLaMA-2-7b-hf",
            "meta-llama/Llama-2-7b-hf",
            #"meta-llama/Llama-2-7b-chat-hf",
            #"meta-llama/Llama-2-13b-hf",
            #"meta-llama/Llama-2-13b-chat-hf"
        ],
        "quantization": "int8",
        #"batch_size": 32,
        "batch_size": 10, # this batch_size is for running inference on the models
        "max_length": 512,
        "top_k": 10,
        "features": [
            "observed",
            "entropy",
            "most_likely",
        ],
        "cache_dir": std.format("./cache/%s", run_name),
        "merge_tokens": false
    },
    "model_params": {
        "num_labels": 2,
        "d_model": 128,
        "n_head": 4,
        "dim_feedforward": 64,
        "n_layers": 1
    },
    "training_params": {
        "output_dir": std.format("./checkpoints_tmp2/%s/training", run_name),
        "save_total_limit": 1,
        "per_device_train_batch_size": 32, # batch_size for the classifier being trained
        "num_train_epochs": 100,
        #"fp16": true,
        "logging_steps": 1,
        "learning_rate": 0.001,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": true,
        "greater_is_better": true,
        "metric_for_best_model": "accuracy",
        "patience": 1000,
    },
    "inference_params": {
        "per_device_eval_batch_size": 32,
        "output_dir": std.format("./checkpoints_tmp2/%s/inference", run_name),
    }
};

{
    "run_name": run_name,
    "train": train_data,
    "test": test_data,
    "model": model,
    "dev": dev_data,
}
