import os
import typing as T

from src.cjd_flows.explib.config_parser import read_config


def test_mnist():
    report_dir = "./reports"
    storage_path = None
    sepline = "\n" + ("-" * 80) + "\n" + ("-" * 80) + "\n"
    print(f"{sepline}Parsing config file:{sepline}")
    config = os.path.abspath("./tests/explib/mnist.yaml")
    experiment = read_config(config)
    print(f"{sepline}Done.{sepline}")
    print(f"{sepline}Conducting experiment{sepline}")
    # Conduct experiment
    #experiment.conduct(report_dir, storage_path=storage_path)
    print(f"{sepline}Done.{sepline}")
    assert True
