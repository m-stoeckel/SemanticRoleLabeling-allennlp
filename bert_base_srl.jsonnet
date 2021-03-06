{
	"dataset_reader": {
		"type": "conll2009_srl",
		"bert_model_name": "bert-base-multilingual-cased"
	},
	//	"iterator": {
	//		"type": "bucket",
	//		"batch_size": 16,
	//		"sorting_keys": [
	//			[
	//				"tokens",
	//				"num_tokens"
	//			]
	//		]
	//	},
	"data_loader": {
		"batch_sampler": {
			"type": "bucket",
			"batch_size": 16
		}
	},
	"train_data_path": "/resources/corpora/CoNLL/CoNLL-2009_Shared_Task/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-train.txt",
	"validation_data_path": "/resources/corpora/CoNLL/CoNLL-2009_Shared_Task/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.txt",
	"model": {
		"type": "srl_bert",
		"embedding_dropout": 0.1,
		"bert_model": "bert-base-multilingual-cased"
	},
	"distributed": {
		"cuda_devices": [
			0,
			1
		]
	},
	"trainer": {
		"optimizer": {
			"type": "huggingface_adamw",
			"lr": 5e-5,
			"correct_bias": false,
			"weight_decay": 0.01,
			"parameter_groups": [
				[
					[
						"bias",
						"LayerNorm.bias",
						"LayerNorm.weight",
						"layer_norm.weight"
					],
					{
						"weight_decay": 0.0
					}
				]
			]
		},
		"learning_rate_scheduler": {
			"type": "slanted_triangular",
			"num_epochs": 15,
			"num_steps_per_epoch": 1125
		},
		"grad_norm": 1.0,
		"num_epochs": 15,
		"validation_metric": "+f1-measure-overall",
		"checkpointer": {
			"num_serialized_models_to_keep": 2,
		},
		"should_log_learning_rate": true
	}
}
