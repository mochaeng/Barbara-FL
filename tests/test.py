from barbara_fl import simulation

train_paths = [
    "./FLdata/client-1: BoT/train.parquet",
    "./FLdata/client-2: UNSW/train.parquet",
    "./FLdata/client-3: CSE/train.parquet",
]

test_paths = [
    "./FLdata/client-1: BoT/test.parquet",
    "./FLdata/client-2: UNSW/test.parquet",
    "./FLdata/client-3: CSE/test.parquet",
]

paths: simulation.Paths = {"train": train_paths, "test": test_paths}

config: simulation.Config = {
    "num_rounds": 10,
    "num_clients": 3,
    "data_paths": paths,
    "algorithm": simulation.Algorithm.FedAVG,
}

optional: simulation.ConfigOptional = {
    "columns_to_remove": [
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "Attack",
    ]
}

print(simulation.Simulation(config, config_optional=optional).start())
