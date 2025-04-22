# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1, 2022/7/6
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan, Gaowei Zhang
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn, zgw15630559577@163.com

"""
recbole.data.utils
########################
"""

import copy
import importlib
import os
import pickle

from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir, get_local_time, set_color
from recbole.utils.argument_list import dataset_arguments
from recbole.sampler import DICESampler


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module("recbole.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.GENERAL: "Dataset",
            ModelType.SEQUENTIAL: "SequentialDataset",
            ModelType.CONTEXT: "Dataset",
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
            ModelType.TRADITIONAL: "Dataset",
            ModelType.DECISIONTREE: "Dataset",
            ModelType.DEBIAS: "Dataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(
        config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth'
    )
    file = config["dataset_save_path"] or default_file
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config["checkpoint_dir"])
    save_path = config["checkpoint_dir"]
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into", "pink") + f": [{file_path}]")
    Serialization_dataloaders = []
    for dataloader in dataloaders:
        generator_state = dataloader.generator.get_state()
        dataloader.generator = None
        dataloader.sampler.generator = None
        Serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(Serialization_dataloaders, f)


def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    default_file = os.path.join(
        config["checkpoint_dir"],
        f'{config["dataset"]}-for-{config["model"]}-dataloader.pth',
    )
    dataloaders_save_path = config["dataloaders_save_path"] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, "rb") as f:
        dataloaders = []
        for data_loader, generator_state in pickle.load(f):
            generator = torch.Generator()
            generator.set_state(generator_state)
            data_loader.generator = generator
            data_loader.sampler.generator = generator
            dataloaders.append(data_loader)

        train_data, valid_data, test_data = dataloaders
    for arg in dataset_arguments + ["seed", "repeatable", "eval_args"]:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(
        set_color("Load split dataloaders from", "pink")
        + f": [{dataloaders_save_path}]"
    )
    return train_data, valid_data, test_data


import random
def introduce_label_noise(dataset, noise_ratio=0):
    """向训练数据中引入标签噪声

    Args:
        dataset: 训练数据加载器
        noise_ratio: 噪声比例，默认为5%

    Returns:
        添加了噪声的训练数据加载器
    """
    if noise_ratio == 0:
        return dataset
    logger = getLogger()
    logger.info(f"正在引入 {noise_ratio * 100}% 的标签噪声...")

    inter_feat = dataset.inter_feat

    # 获取交互数据的副本
    interaction_data = inter_feat.interaction

    # 获取用户和物品字段名
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    # 获取数据集中所有存在的用户ID和物品ID
    all_uids = torch.unique(interaction_data[uid_field])
    all_iids = torch.unique(interaction_data[iid_field])

    # 计算需要替换的样本数量
    total_samples = len(interaction_data[uid_field])
    logger.info(f"共有 {total_samples} 个交互记录")
    num_to_replace = int(total_samples * noise_ratio)

    # 随机选择要替换的样本索引
    if num_to_replace > 0:
        replace_indices = torch.randperm(total_samples)[:num_to_replace]

        # 随机决定是替换用户还是物品
        for idx in replace_indices:
            if random.random() < 0.5:  # 50%概率替换用户
                # 从数据集中已存在的用户ID中随机选择一个不同的用户ID
                original_uid = interaction_data[uid_field][idx]
                valid_uids = all_uids[all_uids != original_uid]
                if len(valid_uids) > 0:
                    new_uid = valid_uids[torch.randint(0, len(valid_uids), (1,))]
                    interaction_data[uid_field][idx] = new_uid
            else:  # 50%概率替换物品
                # 从数据集中已存在的物品ID中随机选择一个不同的物品ID
                original_iid = interaction_data[iid_field][idx]
                valid_iids = all_iids[all_iids != original_iid]
                if len(valid_iids) > 0:
                    new_iid = valid_iids[torch.randint(0, len(valid_iids), (1,))]
                    interaction_data[iid_field][idx] = new_iid

        logger.info(f"已随机替换 {num_to_replace} 个交互记录中的用户或物品ID")

    # 更新数据集的交互特征
    dataset.inter_feat.interaction = interaction_data

    return dataset


def data_preparation(config, dataset, noise_ratio=0.0):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        model_type = config["MODEL_TYPE"]
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        # 引入标签噪声（5%）
        train_dataset = introduce_label_noise(train_dataset, noise_ratio=0.2)
        built_datasets[0] = train_dataset
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, shuffle=config["shuffle"]
            )
        else:
            kg_sampler = KGSampler(
                dataset,
                config["train_neg_sample_args"]["distribution"],
                config["train_neg_sample_args"]["alpha"],
            )
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, kg_sampler, shuffle=True
            )

        valid_data = get_dataloader(config, "evaluation")(
            config, valid_dataset, valid_sampler, shuffle=False
        )
        test_data = get_dataloader(config, "evaluation")(
            config, test_dataset, test_sampler, shuffle=False
        )
        if config["save_dataloaders"]:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data)
            )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        "MacridVAE": _get_AE_dataloader,
        "CDAE": _get_AE_dataloader,
        "ENMF": _get_AE_dataloader,
        "RaCT": _get_AE_dataloader,
        "RecVAE": _get_AE_dataloader,
        "DICE": _get_DICE_dataloader,
    }

    if config["model"] in register_table:
        return register_table[config["model"]](config, phase)

    model_type = config["MODEL_TYPE"]
    if phase == "train":
        if model_type != ModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_mode = config["eval_args"]["mode"]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader

def _get_DICE_dataloader(config, phase):
    """Customized function for DICE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == 'train':
        return DICEDataloader
    else:
        eval_mode = config["eval_args"]["mode"]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _get_AE_dataloader(config, phase):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == "train":
        return UserDataLoader
    else:
        eval_mode = config["eval_args"]["mode"]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ["train", "valid", "test"]
    train_neg_sample_args = config["train_neg_sample_args"]
    eval_neg_sample_args = config["eval_neg_sample_args"]
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args["distribution"] != "none":
        if config['model'] == 'DICE':
            sampler = DICESampler(phases, built_datasets, train_neg_sample_args['distribution'], train_neg_sample_args["alpha"])
        elif not config["repeatable"]:
            sampler = Sampler(
                phases,
                built_datasets,
                train_neg_sample_args["distribution"],
                train_neg_sample_args["alpha"],
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                train_neg_sample_args["distribution"],
                train_neg_sample_args["alpha"],
            )
        train_sampler = sampler.set_phase("train")

    if eval_neg_sample_args["distribution"] != "none":
        if sampler is None:
            if not config["repeatable"]:
                sampler = Sampler(
                    phases,
                    built_datasets,
                    eval_neg_sample_args["distribution"],
                    train_neg_sample_args["alpha"],
                )
            else:
                sampler = RepeatableSampler(
                    phases,
                    dataset,
                    eval_neg_sample_args["distribution"],
                    train_neg_sample_args["alpha"],
                )
        else:
            sampler.set_distribution(eval_neg_sample_args["distribution"])
        valid_sampler = sampler.set_phase("valid")
        test_sampler = sampler.set_phase("test")

    return train_sampler, valid_sampler, test_sampler
