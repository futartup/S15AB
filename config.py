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
                "mean": [0.42969442521673096, 0.4398552525513781, 0.4328128029756167] ,
                "std":[0.28715867313017185, 0.26689101668717086, 0.2673114311850263],
                "always_apply": true
            }
        ]
        },
        "test": {
            "which": "albumentation",
            "what": [
            {
                "name": "Normalize",
                "mean": [0.43169562107826576, 0.44226087717112156, 0.43416351285063187] ,
                "std":[0.28766322395057514, 0.2674202933537084, 0.2684091646207769],
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
