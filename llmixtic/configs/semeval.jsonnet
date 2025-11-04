local run_name = "semeval";

local train_data = {
    "mtdetect_train": {
        dataset_path: "/home/cgarcia/Documentos/mtdetect/mtdetect/llmixtic/subtaskA_train_monolingual.jsonl",
        split: "train",
    },
};

local dev_data = {
    "mtdetect_dev": {
        dataset_path: "/home/cgarcia/Documentos/mtdetect/mtdetect/llmixtic/subtaskA_dev_monolingual.jsonl",
        split: "dev",
    },
};

local test_data = {
    "mtdetect_test": {
        dataset_path: "/home/cgarcia/Documentos/mtdetect/mtdetect/llmixtic/subtaskA_test_monolingual.jsonl",
        split: "test",
    },
};

local model = {
    "feature_params": {
        "models": [
#            "gpt2",
#            "gpt2-medium",
#            "distilgpt2",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-13b-chat-hf"
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
        "output_dir": std.format("./checkpoints/%s/training", run_name),
        "save_total_limit": 1,
        "per_device_train_batch_size": 32, # batch_size for the classifier being trained
#        "num_train_epochs": 10000,
        "num_train_epochs": 10,
        "fp16": true,
        "logging_steps": 50,
        "learning_rate": 0.001,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": true,
        "greater_is_better": true,
        "metric_for_best_model": "accuracy",
        "patience": 3,
#        "patience": 6,
#        "patience": 100,
#        "lr_scheduler_type": "inverse_sqrt",
        "lr_scheduler_type": "linear",
    },
    "inference_params": {
        "per_device_eval_batch_size": 32,
        "fp16": true,
        "output_dir": std.format("./checkpoints/%s/inference", run_name),
    }
};

{
    "run_name": run_name,
    "train": train_data,
    "test": test_data,
    "model": model,
    "dev": dev_data,
}
