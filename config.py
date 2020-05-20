{
    "seed": 1,
    "data": "custom",
    "mean": [0.43072554273179153, 0.4403776258001769, 0.4335964820201932] ,
    "std": [0.28761761089190296, 0.2675504910801714, 0.2683254497043513],
    "num_workers": 2,
    "shuffle": true,
    "epochs": 50,
    "multiclass": true,
    "batch_size": 30,
    "num_workers": 2,
    "lr_step_size": 25,
    "lr_gamma": 0.001,
    "loss": "CrossEntropyLoss",
    "lr_finder": {
        "optimizer": {"lr": 1e-5},
        "range_test": {"end_lr":100, "step_mode": "exp"}
    },
    "optimizer": {
        "type": "Adam",
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
        "bilinear": false
    }
}
