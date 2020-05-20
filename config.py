{
    "seed": 1,
    "data": "custom",
    "num_workers": 2,
    "shuffle": true,
    "epochs": 50,
    "multiclass": true,
    "batch_size": 10,
    "num_workers": 2,
    "lr_step_size": 25,
    "lr_gamma": 0.001,
    "loss": "BCEWithLogitsLoss",
    "lr_finder": {
        "optimizer": {"lr": 1e-5},
        "range_test": {"end_lr":100, "step_mode": "exp"}
    },
    "optimizer": {
        "type": "RMSprop",
        "weight_decay": 1e-8,
        "momentum": 0.9
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "mode": "min",
        "patience": 2
    },
    "lr_finder_use": true,
    "transformations": {
        "train": {
            "which": "albumentation",
            "what": [
            {
                "name": "Cutout",
                "num_holes":1,
                "max_h_size": 80, 
                "max_w_size": 80
            },
            {
                    "name": "Normalize",
                    "mean": [0.42969, 0.43985, 0.43281] ,
                    "std":[0.28715, 0.26689, 0.26731],
                    "always_apply": true
                }
        ]
        },
        "test": {
            "which": "albumentation",
            "what": [
            {
                    "name": "Normalize",
                    "mean": [0.42969, 0.43985, 0.43281] ,
                    "std":[0.28715, 0.26689, 0.26731],
                    "always_apply": true
                }
            ]
        }
    },
    "model": "unet",
    "model_initializer": {
        "n_channels": 3,
        "n_classes": 1,
        "bilinear": true
    },
    "log_dir": "/content/drive/My Drive/Colab Notebooks/S15AB/runs/MDME"
}
