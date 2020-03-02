{

    "dataset_reader": {
        "type": "conll2009_srl",
        "bert_model_name": "bert-base-multilingual-cased"
      },

    "iterator": {
        "type": "bucket",
        "batch_size": 32,
        "sorting_keys": [["tokens", "num_tokens"]]
    },

    "train_data_path": "/resources/corpora/CoNLL/CoNLL-2009_Shared_Task/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-train.txt",
    "validation_data_path": "/resources/corpora/CoNLL/CoNLL-2009_Shared_Task/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.txt",

    "model": {
        "type": "srl_bert",
        "embedding_dropout": 0.1,
        "bert_model": "bert-base-multilingual-cased",
    },

    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 15,
            "num_steps_per_epoch": 8829,
        },
        "grad_norm": 1.0,
        "num_epochs": 15,
        "validation_metric": "+f1-measure-overall",
        "num_serialized_models_to_keep": 2,
        "should_log_learning_rate": true,
        "cuda_device": "0,1"
    },

}
