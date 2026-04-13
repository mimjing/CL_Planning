'''自动化评测脚本'''
import os
import re
import subprocess

from ruamel.yaml import YAML

# ====== 配置路径 ======
CONFIG_PATH = "/data_set/UniTraj/unitraj/configs/config.yaml"
# ====== 方法 ↔ yaml 文件映射 ======
method_yaml_map = {
    "autobot": "/data_set/UniTraj/unitraj/configs/method/autobot.yaml",
    "MTR":     "/data_set/UniTraj/unitraj/configs/method/MTR.yaml",
    "VBD":     "/data_set/UniTraj/unitraj/configs/method/VBD.yaml",
    "wayformer": "/data_set/UniTraj/unitraj/configs/method/wayformer.yaml",
}
EVAL_SCRIPT = "python acc_sim.py"

# ====== 实验列表 ======
exp_list = [
    "autobot_waymo-waymo",
    "autobot_nuplan-waymo",
    "autobot_waymo-merge",
    "autobot_nuplan-merge",
    "autobot_waymo-nuplan",
    "autobot_nuplan-nuplan",
    # "MTR_merge-real",
    # "MTR_nuplan-real",
    # "MTR_waymo-real",
]

# ====== ckpt 路径映射 ======
# ====== 自动 ckpt 映射 ======
CKPT_ROOT = "/data_set/UniTraj/unitraj/outputs"

ckpt_map = {
    "autobot": {
        "merge": f"{CKPT_ROOT}/autobot_merge/brier_fde=2.14.ckpt",
        "nuplan": f"{CKPT_ROOT}/autobot_nuplan/brier_fde=1.85.ckpt",
        "waymo": f"{CKPT_ROOT}/autobot_waymo/brier_fde=2.11.ckpt",
    },
    "MTR": {
        "merge": f"{CKPT_ROOT}/MTR_merge/brier_fde=1.99.ckpt",
        "nuplan": f"{CKPT_ROOT}/MTR_nuplan/brier_fde=1.62.ckpt",
        "waymo": f"{CKPT_ROOT}/MTR_waymo/brier_fde=1.98.ckpt",
    },
    "VBD": {
        "merge": f"{CKPT_ROOT}/VBD_merge/epoch=14.ckpt",
        "nuplan": f"{CKPT_ROOT}/VBD_nuplan/epoch=15.ckpt",
        "waymo": f"{CKPT_ROOT}/VBD_waymo/epoch=15.ckpt",
    },
    "wayformer": {
        "merge": f"{CKPT_ROOT}/wayformer_merge/brier_fde=2.68.ckpt",
        "nuplan": f"{CKPT_ROOT}/wayformer_nuplan/brier_fde=2.23.ckpt",
        "waymo": f"{CKPT_ROOT}/wayformer_waymo/brier_fde=2.71.ckpt",
    },
}

def replace_line(file_path, key, new_value):
    """仅替换 yaml 文件中 key 对应的行"""
    with open(file_path, "r") as f:
        lines = f.readlines()

    pattern = re.compile(rf"^\s*{key}\s*:")
    found = False
    for i, line in enumerate(lines):
        if pattern.match(line):
            indent = re.match(r"^(\s*)", line).group(1)
            lines[i] = f"{indent}{key}: {new_value}\n"
            found = True
            break

    if not found:
        print(f"⚠️ Warning: key '{key}' not found in {file_path}")
    else:
        with open(file_path, "w") as f:
            f.writelines(lines)

def replace_defaults_method(yaml_path, new_method):
    """只改 defaults 列表里的 method: xxx"""
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yaml_path) as f:
        data = yaml.load(f)

    # defaults 是列表，找到 item 以 "method:" 开头
    for i, item in enumerate(data.get("defaults", [])):
        if isinstance(item, dict) and "method" in item:
            item["method"] = new_method  # 直接改值
            break
    else:
        print("⚠️ defaults 列表里找不到 dict 形式的 method 项")

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

for exp_name in exp_list:
    print(f"\n🚀 开始处理实验：{exp_name}")

    # --- 解析实验名 ---
    method_prefix = exp_name.split("_")[0]          # autobot / MTR / ...
    middle        = exp_name.split("_")[1].split("-")[0]
    last          = exp_name.split("-")[-1]

    # --- 1. 改 config.yaml：method + exp_name + data_path ---
    replace_defaults_method(CONFIG_PATH, method_prefix)   # 放你循环里
    replace_line(CONFIG_PATH, "exp_name", f'"{exp_name}"')
    replace_line(CONFIG_PATH, "eval_data_path", f'"/data_set/UniTraj/unitraj/eval/{last}"')
    # 刷盘
    with open(CONFIG_PATH, "r+") as f:
        f.flush(); os.fsync(f.fileno())

    # --- 2. 定位对应 method yaml ---
    method_yaml = method_yaml_map[method_prefix]
    ckpt_path = ckpt_map[method_prefix].get(middle)
    if not ckpt_path:
        print(f"⚠️ 未定义 {middle} 的 ckpt 路径，跳过该实验")
        continue

    replace_line(method_yaml, "ckpt_path", f'"{ckpt_path}"')
    with open(method_yaml, "r+") as f:
        f.flush(); os.fsync(f.fileno())
    print(f"✅ 修改 {method_prefix}.yaml: ckpt={ckpt_path}")

    # --- 3. 运行评测 ---
    print(f"▶️ 开始执行评测脚本...")
    try:
        subprocess.run(EVAL_SCRIPT, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 评测出错: {e}")
        continue

    print(f"🎯 实验 {exp_name} 评测完成。")