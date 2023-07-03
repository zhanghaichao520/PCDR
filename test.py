# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
from logging import getLogger
from collections import Counter, defaultdict
import sys
import pandas as pd
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
import numpy as np
import os
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
)
from recbole.utils.url import (
    makedirs,
)

def copy_file(config, source_dataset_path, target_dataset_path, token, file_type):
    filepath = os.path.join(source_dataset_path, f"{token}.{file_type}")
    if not os.path.isfile(filepath):
        return
    field_separator = config["field_separator"]
    encoding = config["encoding"]
    df = pd.read_csv(
        filepath,
        delimiter=field_separator,
        encoding=encoding,
        engine="python",
    )
    df.to_csv(os.path.join(target_dataset_path, f"{token}.{file_type}"), sep=config["seq_separator"], header=True,
                      index=False)
def split_dataset(config):
    token = config["dataset"]
    dataset_path = config["data_path"]

    filepath = os.path.join(dataset_path, f"{token}.inter")
    if not os.path.isfile(filepath):
        raise ValueError(f"File {filepath} not exist.")
    field_separator = config["field_separator"]
    encoding = config["encoding"]
    df = pd.read_csv(
        filepath,
        delimiter=field_separator,
        encoding=encoding,
        engine="python",
    )

    split_radio = config["eval_args"]["split"]["RS"][2]

    test_part = df.sample(frac=split_radio)
    test_part.reset_index(drop = True,inplace = True)

    # add interaction_num_countdown for sample test dataset in build method
    iid_field = config["USER_ID_FIELD"]+":token"
    colname_interaction_num_countdown = "interaction_num_countdown"
    item_inter_num = Counter(test_part[iid_field].values)

    inter_fre_list = []
    for item in test_part[iid_field]:
        if item_inter_num[item] == 0:
            inter_fre_list.append(0)
        else:
            inter_fre_list.append(1 / item_inter_num[item])
    test_part[colname_interaction_num_countdown] = inter_fre_list
    # test_part = test_part.sample(frac=0.2, replace=False)
    test_part = test_part.sample(frac=0.2, replace=False, weights='interaction_num_countdown')

    test_data_total_num = len(test_part)
    print("测试集 数据量" + f"[{test_data_total_num}]" )
    print("测试集 item交互频率最大值" + f"[{np.max(list(item_inter_num.values()))}]" )
    print("测试集 item交互频率最小值" + f"[{np.min(list(item_inter_num.values()))}]" )

    # test_part = test_part[(1 / test_part["interaction_num_countdown"] >= 10) & ( 1 / test_part["interaction_num_countdown"] <= 1000)]
    print("测试集截断后数据量" + f"[{len(test_part)}]")
    print("测试集截取数据占比：" + f"[{len(test_part) / test_data_total_num}]")
    test_part.drop("interaction_num_countdown", axis=1, inplace=True)

    test_file_path = dataset_path + "-test"
    if not os.path.exists(test_file_path):
        makedirs(test_file_path)
    test_part.to_csv(os.path.join(test_file_path, f"{token}.inter"), sep=config["seq_separator"], header=True,
                     index=False)

    train_part = df.drop(test_part.index)

    if config["load_inter_data_limit"] is not None:
        train_part = train_part.sample(n=min(config["load_inter_data_limit"], len(train_part)))

    train_part.reset_index(drop = True,inplace = True)
    train_file_path = dataset_path + "-train"
    if not os.path.exists(train_file_path):
        makedirs(train_file_path)
    train_part.to_csv(os.path.join(train_file_path, f"{token}.inter"), sep=config["seq_separator"], header=True,
                     index=False)

    copy_file(config, dataset_path, test_file_path, token, "user")
    copy_file(config, dataset_path, test_file_path, token, "item")

    copy_file(config, dataset_path, train_file_path, token, "user")
    copy_file(config, dataset_path, train_file_path, token, "item")


    config["field_separator"] = config["seq_separator"]
    config["test_datapath"] = test_file_path
    config["train_datapath"] = train_file_path



def run_recbole(
    model=None, dataset=None, config_file_list=None, config_dict=None, saved=True
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    split_dataset(config)

    config["data_path"] = config["train_datapath"]
    config["eval_args"]["split"]["RS"] = [0.8, 0.2, 0]

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)


    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model_file = "saved/DICE-Jun-09-2023_18-01-18.pth"
    # model_file = "saved/CausE-Jun-09-2023_18-01-23.pth"
    # model_file = "saved/DMCB-Jun-09-2023_16-56-15.pth"
    # model_file = "saved/DCCL-Jul-03-2023_10-41-27.pth"
    model_file = None

    # model training
    if model_file is None:
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config["show_progress"]
        )


    config["data_path"] = config["test_datapath"]
    config["eval_args"]["split"]["RS"] = [0.1, 0, 0.9]

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
        , model_file=model_file
    )

    if model_file is None:
        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="DCCL", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="jester", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    run_recbole(
        model=args.model, dataset=args.dataset, config_file_list=config_file_list
    )