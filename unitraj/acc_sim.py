import sys
import os
import time
import yaml
import argparse
import logging

from tqdm import tqdm
from multiprocessing import Pool

from unitraj.closeloop.VBD.vbd_policy import VBDPolicy
from unitraj.closeloop.unitraj.unitraj_policy import UniTrajPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from unitraj.utils.evaluate_utils import EvaluateMetrics

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

sys.path.append('/home/mj/PycharmProjects/scenarionet-main/src')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.WARNING)

# ===============================
# 加速闭环推理
# 参数与配置
# ===============================
parser = argparse.ArgumentParser(description="Parallel evaluation")
parser.add_argument("--file_path", default='/data_set/UniTraj/unitraj/configs/config.yaml')
parser.add_argument("--max_step", default=80, type=int)
parser.add_argument("--eval_eps", default=800, type=int)
parser.add_argument("--num_workers", default=3, type=int, help="并行进程数")
args = parser.parse_args()

with open(args.file_path, "r") as file:
    cfg = yaml.safe_load(file)
method = cfg['defaults'][0]['method']
database_path = cfg['eval_data_path']

if method in ['wayformer', 'autobot', 'MTR']:
    policy = UniTrajPolicy
elif method == 'VBD':
    policy = VBDPolicy


# ===============================
# 环境创建函数
# ===============================
def create_env(ep_idx):
    env = ScenarioEnv(
        {
            "use_render": False,
            "agent_policy": policy,
            "manual_control": False,
            "log_level": logging.CRITICAL,
            "num_scenarios": 1,  # 每个进程单独运行一个
            "sequential_seed": False,
            "start_scenario_index": ep_idx,
            "horizon": args.max_step,
            "vehicle_config": dict(
                show_navi_mark=False,
                show_line_to_dest=False,
                show_dest_mark=False,
                no_wheel_friction=False,
            ),
            "data_directory": database_path,
        }
    )
    return env


# ===============================
# 单个进程运行的函数
# ===============================
def run_episode(ep_idx):
    """单进程评测一个场景"""
    try:
        os.environ.pop('METADRIVE_ENGINE_INITIALIZED', None)
        from metadrive.engine.engine_utils import close_engine
        close_engine()
        eval_env = create_env(ep_idx)
        evaluate = EvaluateMetrics()
        eval_env.reset()

        total_steps, eps_reward = 0, 0

        for i in range(args.max_step):
            o, r, tm, tc, info = eval_env.step([0, 0])
            evaluate.step(info, o, i, eval_env)
            total_steps += 1
            eps_reward += r
            if tm or tc:
                break

        scene_score, score_list, success = evaluate.reset(total_steps, eval_env)
        mean_reward = eps_reward / total_steps
        print("scene_score:", scene_score)


        result = dict(
            ep_idx=ep_idx,
            scene_score=scene_score,
            round_scores=score_list,
            mean_reward=mean_reward,
            success=success,
            mean_speed=evaluate.total_speed[-1]
            if len(evaluate.total_speed) > 0 else 0,
            I_progress=evaluate.I_round_scores['I_progress'][-1] if evaluate.I_round_scores['I_progress'] else 0,
            I_comfort=evaluate.I_round_scores['I_comfort'][-1] if evaluate.I_round_scores['I_comfort'] else 0,
        )
        eval_env.close()
        return result

    except Exception as e:
        print(f"[⚠️] Episode {ep_idx} 出错: {e}")
        return dict(ep_idx=ep_idx, scene_score=0, round_scores={},
            mean_reward=0.0, success=0, mean_speed=0, I_progress=0, I_comfort=0)


# ===============================
# 主进程聚合逻辑
# ===============================
if __name__ == '__main__':
    t1 = time.time()

    print(f"🚀 启动 {args.num_workers} 个进程并行评测 {args.eval_eps} 个场景...")
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(run_episode, range(args.eval_eps)), total=args.eval_eps))

    # 聚合结果
    valid_results = [r for r in results if r is not None]
    success_flags = [r['success'] for r in valid_results]
    mean_rewards = [r['mean_reward'] for r in valid_results]

    # === 合并 EvaluateMetrics ===
    merged_eval = EvaluateMetrics()
    for r in valid_results:
        for k, v in r['round_scores'].items():
            merged_eval.round_scores[k].append(v)
        merged_eval.I_round_scores['I_progress'].append(r['I_progress'])
        merged_eval.I_round_scores['I_comfort'].append(r['I_comfort'])
        merged_eval.total_speed.append(r['mean_speed'])
        merged_eval.total_scores.append(r['scene_score'])

    # === 计算总体指标 ===
    success_rate = sum(success_flags) / len(success_flags)
    mean_reward_all = sum(mean_rewards) / len(mean_rewards)
    info_eval = {"mean_reward_all": mean_reward_all, "success_rate": success_rate}

    # === 调用 EvaluateMetrics 原生输出 ===
    merged_eval.print_metrics(info_eval, file_path="metrics_real", file_name=cfg['exp_name'])

    print(f"⏱ 总耗时: {time.time() - t1:.1f} 秒")
    print("🎉 评测完成！")
