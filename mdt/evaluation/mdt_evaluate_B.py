from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
import calvin_env

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist
from PIL import Image

from mdt.evaluation.multistep_sequences import get_sequences
from mdt.evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition, join_vis_lang
from mdt.utils.utils import get_last_checkpoint
from mdt.rollout.rollout_video import RolloutVideo

logger = logging.getLogger(__name__)


def get_video_tag(i, scene_name):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    # return f"_long_horizon/sequence_{i}"
    # MODIFIED: saving file name convention
    return f"{scene_name[-1]}_{i:05d}"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, cfg, log_dir=None):
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    sequences = get_sequences(cfg.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        epoch = checkpoint.stem.split("=")[1]
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
        wandb.log({"avrg_performance/avg_seq_len": avg_seq_len, "avrg_performance/chain_sr": chain_sr, "detailed_metrics/task_info": task_info})
        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")


def evaluate_policy(model, env, lang_embeddings, cfg, num_videos=0, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    # CHECK: val_annotations is a dictionary of task -> list of lang annotations,
    # however, all lists are length 1. 
    # {'rotate_red_block_right': ['take the red block and rotate it to the right'], 'rotate_red_block_left': ['take the red block and rotate it to the left'], 'rotate_blue_block_right': ['take the blue block and rotate it to the right'], 'rotate_blue_block_left': ['take the blue block and rotate it to the left'], 
    # CHECK: yaml config 에서 defaults: annotations: new_playtable_validation은 len(1) dict, new_playtable은 length가 긴 dict이므로 바꿔서 해볼 것
    # MODIFIED: for saving convention -> global_step
    global_step = 0

    # video stuff
    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            # CHECK: change video saving directory here
            save_dir=save_dir,
            # CHECK: resolution scale을 192/200으로 설정한다면?
            resolution_scale=1.0,
        )
    else:
        rollout_video = None

    # TODO: sequence generated here, modify numpy seed inside get_sequence function
    eval_sequences = get_sequences(cfg.num_sequences)

    results = []
    plans = defaultdict(list)

    if not cfg.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    num_saved = cfg.dataset_generation.num_saved

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        record = num_saved < num_videos
        # MODIFIED: exit if we have enough videos
        if not record:
            break
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, num_saved, save_dir, global_step
        )
        results.append(result)
        print(num_saved, result)
        if record:
            # NOTE: log_to_file = True이므로 아무것도 안함
            rollout_video.write_to_tmp()
        if not cfg.debug:
            success_rates = count_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)
        if not cfg.dataset_generation.skip_failed or (cfg.dataset_generation.skip_failed and result == 5):
            num_saved += 1
        if record and num_saved and num_saved % cfg.dataset_generation.flush_interval == 0:
            # save videos
            print()
            print("####################################################################")
            print(f"flushing {cfg.dataset_generation.flush_interval} videos")
            print(f"progress {i}/{cfg.num_sequences} done")
            print("####################################################################")
            rollout_video._log_videos_to_file(global_step, save_as_video=True)
            rollout_video.pop_all()
        

    if num_videos > 0:
        # log rollout videos
        # CHECK: have to modify this
        print(f"saving {num_saved % cfg.dataset_generation.flush_interval} videos")
        rollout_video._log_videos_to_file(global_step, save_as_video=True)
    return results, plans


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i, save_dir, global_step
):
    # MODIFIED: initial position of blocks changed according to scene name (e.g. calvin_env_D)
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state, shuffle = cfg.dataset_generation.shuffle_initial, scene=cfg.dataset_generation.scene[-1])
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i, cfg.dataset_generation.scene), caption=caption)
    success_counter = 0
    lang_annotations_idx = []
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        if record:
            rollout_video.new_subtask()
        success = rollout(env, model, task_checker, cfg, subtask, lang_embeddings, val_annotations, record, rollout_video)
        if record:
            # MODIFIED: removed draw_outcome 
            # rollout_video.draw_outcome(success)
            lang_annotations_idx.append([rollout_video.sub_task_beginning,rollout_video.step_counter-1])
            
        if success:
            success_counter += 1
        else:
            # MODIFIED: return success_counter -> break
            break
    # MODIFIED: save lang_annotations seperately
    if record and (cfg.dataset_generation.skip_failed and success_counter==5) or not cfg.dataset_generation.skip_failed:
        lang_annotations = [val_annotations[subtask][0] for subtask in eval_sequence]
        #lang_annotations.append(success_counter)
        save_path_lang = f"{save_dir}/{get_video_tag(i, cfg.dataset_generation.scene).replace('/', '_')}_{global_step}_lang.npy"
        save_path_lang_idx = f"{save_dir}/{get_video_tag(i, cfg.dataset_generation.scene).replace('/', '_')}_{global_step}_lang_idx.npy"
        np.save(save_path_lang, lang_annotations)
        np.save(save_path_lang_idx, lang_annotations_idx)
    if cfg.dataset_generation.skip_failed and not success_counter == 5:
        rollout_video.pop_last()
    return success_counter


def rollout(env, model, task_oracle, cfg, subtask, lang_embeddings, val_annotations, record=False, rollout_video=None):
    with torch.no_grad():
        if cfg.debug:
            print(f"{subtask} ", end="")
            time.sleep(0.5)
        obs = env.get_obs()
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        # get language goal embedding
        goal = lang_embeddings.get_lang_goal(lang_annotation)
        goal['lang_text'] = val_annotations[subtask][0]
        model.reset()
        start_info = env.get_info()

        for step in range(cfg.ep_len):
            action = model.step(obs, goal)
            obs, _, _, current_info = env.step(action)
            if cfg.debug:
                pass
                # img = env.render(mode="rgb_array")
                # CHECK: 왜 join_vis_lang에서 cv2를 open하다가 아무 반응 없이 멈추지?
                # join_vis_lang(img, lang_annotation)
                # time.sleep(0.1)
            if record and step % cfg.dataset_generation.sparsity == 0:
                # update video
                # NOTE: observation RGB per frame
                rollout_video.update(obs["rgb_obs"]["rgb_static"])
                # img = np.clip(((obs["rgb_obs"]["rgb_static"].cpu() / 2) + 0.5) * 255, 0, 255).astype(np.uint8)
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # cv2.imsave("/131_data/jihwan/2025_avdm/calvin_dataset_generation/mdt/frame_{step}.png", img)
                # breakpoint()
                # frame = Image.fromarray((((obs["rgb_obs"]["rgb_static"].squeeze() / 2) +0.5).clamp(0,1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                # frame.save(f"/131_data/jihwan/2025_avdm/calvin_dataset_generation/mdt/frame_{step}.png")
                # breakpoint()
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if cfg.debug:
                    print(colored("success", "green"), end=" ")
                if record:
                    # CHECK: adding lang annotation to video here
                    # rollout_video.add_language_instruction(lang_annotation)
                    pass
                return True
        if cfg.debug:
            # NOTE: 하나라도 실패하면 다음 task로 넘어가지 못하고 360 step을 다 소모함
            print(colored("fail", "red"), end=" ")
        if record:
            # CHECK: adding lang annotation to video here
            # rollout_video.add_language_instruction(lang_annotation)
            pass
        return False


@hydra.main(config_path="../../conf", config_name="mdt_evaluate_B")
def main(cfg):
    log_wandb = cfg.log_wandb
    torch.cuda.set_device(cfg.device)
    # CHECK: change seed to generate new sequences every time
    seed_everything(0, workers=True)  # type:ignore
    # evaluate a custom model
    checkpoints = [get_last_checkpoint(Path(cfg.train_folder))]
    lang_embeddings = None
    env = None
    results = {}
    plans = {}

    for checkpoint in checkpoints:
        print(f'device: {cfg.device}')
        model, env, _, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.dataset_path,
            checkpoint,
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=cfg.device,
            scene=cfg.dataset_generation.scene
        )

        print(cfg.num_sampling_steps, cfg.sampler_type, cfg.multistep, cfg.sigma_min, cfg.sigma_max, cfg.noise_scheduler)
        model.num_sampling_steps = cfg.num_sampling_steps
        model.sampler_type = cfg.sampler_type
        model.multistep = cfg.multistep
        if cfg.sigma_min is not None:
            model.sigma_min = cfg.sigma_min
        if cfg.sigma_max is not None:
            model.sigma_max = cfg.sigma_max
        if cfg.noise_scheduler is not None:
            model.noise_scheduler = cfg.noise_scheduler

        if cfg.cfg_value != 1:
            raise NotImplementedError("cfg_value != 1 not implemented yet")

        model.eval()
        if log_wandb:
            log_dir = get_log_dir(cfg.train_folder)
            os.makedirs(log_dir / "wandb", exist_ok=False)
            run = wandb.init(
                project='calvin_eval',
                entity=cfg.wandb.entity,
                group=cfg.model_name + cfg.sampler_type + '_' + str(cfg.num_sampling_steps) + '_steps_' + str(cfg.cond_lambda) + '_c_' + str(cfg.num_sequences) + '_rollouts_',
                config=dict(cfg),
                dir=log_dir / "wandb",
            )
            if cfg.dataset_generation.save_dir:
                save_dir = Path(cfg.dataset_generation.save_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
                os.makedirs(save_dir, exist_ok=False)
            else:
                save_dir = Path(log_dir)
            print(f"saving to {save_dir}")

            results[checkpoint], plans[checkpoint] = evaluate_policy(model, env, lang_embeddings, cfg, num_videos=cfg.num_videos, save_dir=save_dir)

            # just print success rates, etc. videos are not saved here
            print_and_save(results, plans, cfg, log_dir=log_dir)
            run.finish()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main()
