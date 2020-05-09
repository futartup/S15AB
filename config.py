{
    "seed": 1,
    "data": "custom",
    "mean": [0.43050, 0.44021, 0.43348],
    "std": [0.28751, 0.26746, 0.26826],
    "num_workers": 2,
    "shuffle": true,
    "epochs": 50,
    "multiclass": true,
    "batch_size": 512,
    "num_workers": 2,
    "lr_step_size": 25,
    "lr_gamma": 0.001,
    "loss": "CrossEntropyLoss",
    "lr_finder": {
        "optimizer": {"lr": 1e-5},
        "range_test": {"end_lr":100, "step_mode": "exp"}
    },
    "optimizer": {
        "type": "SGD",
        "weight_decay": 1e-4,
        "nesterov": true,
        "momentum": 0.9
    },
    "scheduler": {
        "type": "OneCycleLR",
        "pct_start": 0.2,
        "anneal_strategy": "linear",
        "cycle_momentum": true,
        "base_momentum": 0.80,
        "max_momentum": 0.85
    },
    "lr_finder_use": true,
    "transformations": {
        "train": {
            "which": "albumentation",
            "what": [
            {
                "name": "PadIfNeeded",
                "min_height": 40,
                "border_mode": 4,
                "p": 1.0
            },
            {
                "name": "RandomCrop",
                "height": 32
            },
            {
                "name": "HorizontalFlip"
            },
            {
                "name": "Cutout",
                "num_holes":1,
                "max_h_size": 8, 
                "max_w_size": 8
            },
            {
                "name": "Normalize",
                "mean": [0.43050],
                "std": [0.28751],
                "always_apply": true
            }

        ]
        },
        "test": {
            "which": "albumentation",
            "what": [
                {
                 "name": "Normalize",
                 "mean": [0.43050],
                 "std": [0.28751],
                 "always_apply": true
                }
            ]
        }
    },
    "model": "unet",
    "model_initializer": {
        "n_channels": 1,
        "n_classes": 2,
        "bilinear": false
    }
}
