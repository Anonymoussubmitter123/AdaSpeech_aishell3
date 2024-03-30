import argparse

import yaml

from preprocessor import aishell3


def main(config):
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "finetune" in config["dataset"]:
        aishell3.prepare_align(config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
"""
这段代码是一个预处理脚本的入口程序，通过读取指定路径的配置文件，调用不同数据集的预处理程序对音频和文本进行对齐（alignment）处理。
具体来说，它根据配置文件中的 "dataset" 字段指定需要处理的数据集，然后分别调用 ljspeech、aishell3 和 libritts 中的 prepare_align
函数进行处理。这里使用 argparse 模块对命令行参数进行解析，并使用 yaml 模块读取配置文件。
"""