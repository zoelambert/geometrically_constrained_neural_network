config = {
    "dataset": {
        "data_dir": "",
        "train_dir": "train",
        "validation_dir": "validation",
        "batch_size": 5,
    },
    "train": {
        "learning_Rate": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "checkpoint_dir": "",
        "mu": 0.5,
        "tau": 0.4,
        "sigma": 0.01,
        "nb_epoch": 150,
    },
    "primal_dual": {
        "nb_iter": 80,
    },
}
