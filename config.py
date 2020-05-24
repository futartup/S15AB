{
    "seed": 1,
    "lr": 0.001,
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
        "type": "Adam",
        "weight_decay": 1e-4
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
                "max_w_size": 80,
                "always_apply": true
            },
            {
                "name": "Blur",
                "always_apply": false,
                "blur_limit"= (3,7),
                "p": 0.2
            },
            {
                "name": "Solarize",
                "always_apply": false,
                "threshold": (128, 128),
                "p": 0.2
            },
            {
                "name": "GaussNoise",
                "always_apply": false,
                "p": 0.2,
                "var_limit": (10.0, 291.9499816894531)                
            },
            {
                "name": "Normalize",
                "mean": [0.429694425216733, 0.43985525255137686, 0.43281280297561686] ,
                "std":[0.28715867313016924, 0.266891016687173, 0.26731143118502665],
                "always_apply": true
            }
        ]
        },
        "test": {
            "which": "albumentation",
            "what": [
                {
                    "name": "Normalize",
                    "mean": [0.4316956210782655, 0.4422608771711209, 0.43416351285063187],
                    "std": [0.2876632239505755, 0.26742029335370965, 0.26840916462077713],
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
