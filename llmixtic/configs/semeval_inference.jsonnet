local run_name = "semeval";

local test_data = {
    "mtdetect_test": {
        dataset_path: "/home/cgarcia/Documentos/mtdetect/mtdetect/llmixtic/subtaskA_test_monolingual.jsonl",
        split: "test",
    },
};

#local language = "de-en";
#local mt = "Unbabel_TowerInstruct-7B-v0.2";
#local run_name = std.format("mtdetect_%s_%s", [mt, language]);
#
#local test_data = {
#    "mtdetect_test": {
#        dataset_path: std.format("/home/cgarcia/Documentos/mtdetect/mtdetect/wmt_data/%s-mtpaper.all.sentences.shuf.%s.%s.out.all.shuf", [language, "test", mt]),
#        split: "test",
#    },
#};

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
    "training_params": null,
    "inference_params": {
        "per_device_eval_batch_size": 32,
        "fp16": true,
        "output_dir": std.format("./checkpoints/%s/inference", run_name),
        #"output_dir": std.format("./checkpoints_qweqwe/%s/inference", run_name), # using data from mtdetect
    }
};

{
    "run_name": run_name,
    "train": null,
    "test": test_data,
    "model": model,
    "dev": null,
}
