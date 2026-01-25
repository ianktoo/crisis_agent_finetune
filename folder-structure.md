crisis_agent_finetune/
│
├── configs/
│   ├── training_config.yaml
│   ├── model_config.yaml
│   └── dataset_config.yaml
│
├── data/
│   └── local_cache/        # optional: tokenized dataset cache
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_dataset.py
│   │   └── format_records.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── load_model.py
│   │   └── apply_lora.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluation.py
│   │
│   └── utils/
│       ├── logging.py
│       └── json_validator.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── merge_lora.py
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── final_model/
│
├── README.md
└── requirements.txt