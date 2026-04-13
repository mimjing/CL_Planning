import sys
import os
import time
import traceback

import yaml
from tqdm import tqdm
from metadrive.envs.scenario_env import ScenarioEnv

from unitraj.utils.evaluate_utils import EvaluateMetrics
from unitraj.openloop.VBD.vbd_policy import OpenVBDPolicy
from unitraj.openloop.unitraj.unitraj_policy import OpenUniTrajPolicy

sys.path.append('/home/mj/PycharmProjects/scenarionet-main/src')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import logging
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.WARNING)

desc = "Load a database to simulator and replay scenarios"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--file_path", default='/data_set/UniTraj/unitraj/configs/config.yaml', help="The class of the model")
parser.add_argument("--render", default="2D", choices=["none", "2D", "3D", "advanced", "semantic"])
parser.add_argument("--max_step", default=80, type=int)
parser.add_argument("--eval_eps", default=26, type=int)
args = parser.parse_args()

with open(args.file_path, "r") as file:
    cfg = yaml.safe_load(file)
method = cfg['defaults'][0]['method']
database_path = cfg['eval_data_path']

if method in ['wayformer', 'autobot', 'MTR']:
    policy = OpenUniTrajPolicy
elif method == 'VBD':
    policy = OpenVBDPolicy

def create_env():
    env = ScenarioEnv(
        {
            "use_render": False,
            "agent_policy": policy,
            "manual_control": False,
            "log_level": logging.CRITICAL,
            "num_scenarios": 1600,
            "sequential_seed": True,
            "start_scenario_index": 0,
            "horizon": 80,
            "vehicle_config": dict(
                show_navi_mark=True,
                show_line_to_dest=True,
                show_dest_mark=True,
                no_wheel_friction=False,
            ),
            "data_directory": database_path,
        }
    )
    return env



if __name__ == '__main__':
    eval_env = create_env()
    scene_score = []
    success_list = []
    success_rate = 0
    num_epoch = 0
    total_steps = 0
    total_reward = 0
    mean_reward_all = 0
    evaluate = EvaluateMetrics()
    # used_ids = set()

    pbar = tqdm(range(args.eval_eps), desc="Eval eps", unit="ep")
    t1 = time.time()
    try:
        for i_ep in pbar:

            eval_env.reset()

            scenario = eval_env.engine.data_manager.current_scenario
            sdc_id = scenario['metadata']['sdc_id']
            # if scenario['id'] in used_ids or scenario['length'] < 81 or scenario['metadata']['object_summary'][sdc_id][
            #     'moving_distance'] < 10:
            #     # print('跳过',scenario['id'] in used_ids, scenario['length'] < 81, scenario['metadata']['object_summary'][sdc_id]['moving_distance'] < 10)
            #     continue

            # used_ids.add(scenario['id'])

            # eval_env.head_renderer = HeadTopDownRenderer(eval_env)
            i_step = 0
            eps_reward = 0
            num_epoch += 1
            eval_env.engine.agents['default_agent'].expert_traj = scenario['tracks'][sdc_id]['state']["position"]
            # print('id',scenario['id'])

            for i in range(args.max_step):

                o, r, tm, tc, info = eval_env.step([0, 0])
                evaluate.step(info, o, i, eval_env)

                # eval_env.head_renderer.render(
                #     screen_record=False,
                #     show_plan_traj=True,
                #     scaling=6,
                #     film_size=(6000, 4000),
                #     mode="topdown",
                #     text={'步数':i,
                #           'id':scenario['id']},
                # )

                i_step += 1
                total_steps += 1
                total_reward += r
                eps_reward += r

                if tm or tc:
                    score, final_score, _ = evaluate.reset(total_steps, eval_env)
                    scene_score.append(score)
                    print(final_score)
                    average_score = sum(scene_score) / len(scene_score)
                    assert len(scene_score) == num_epoch, '场景评分与回合数量不对应'

                    mean_reward_all = total_reward / total_steps
                    mean_reward = eps_reward / i_step
                    print("scene_score:", scene_score[num_epoch - 1])
                    print("average_score:", average_score)
                    print('——————————————————————————————')
                    # print("整体reward per step:", mean_reward_all)
                    # print("当前回合reward per step:", mean_reward)
                    if not (info['crash_object'] or info['crash_vehicle'] or info['crash_human'] or info[
                        'crash_building'] or info['crash_sidewalk']):
                        success_list.append(1)
                    else:
                        success_list.append(0)
                    success_rate = sum(success_list) / len(success_list)

                    pbar.set_postfix({'success_rate': f"{success_rate:.3f}"})
                    break
        info_eval = {'mean_reward_all': mean_reward_all, 'success_rate': success_rate}
        evaluate.print_metrics(info_eval, file_name=cfg['exp_name'])
        t2 = time.time()
        print('耗时：', t2-t1)
    except (KeyboardInterrupt, Exception) as e:
        print('! ! ! ! ! ! ! ! !')
        print('运行时出错：', e)
        traceback.print_exc()
        logging.exception("Exception in evaluation loop")
        info_eval = {'mean_reward_all': mean_reward_all, 'success_rate': success_rate}
        evaluate.print_metrics(info_eval, file_name=cfg['exp_name'])
        t2 = time.time()
        print('耗时：', t2 - t1)