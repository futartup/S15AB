{
    "seed": 1,
    "data": "custom",
    "num_workers": 2,
    "shuffle": true,
    "epochs": 20,
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
        "weight_decay": 1e-4,
        "momentum": 0.9
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 2
    },
    "lr_finder_use": false,
    "transformations": {
        "train": {
            "which": "albumentation",
            "what": [
            {
                "name": "Cutout",
                "num_holes":1,
                "max_h_size": 80, 
                "max_w_size": 80
            }
        ]
        },
        "test": {
            "which": "albumentation",
            "what": [
          
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
