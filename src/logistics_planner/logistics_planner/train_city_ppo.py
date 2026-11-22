"""Fine tune direct PPO on city geometry, then test in the real city simulator."""
import argparse
import json
import time
import shutil
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from .city_ppo_env import CityPPOEnv
from .direct_ppo_policy import transfer_policy_weights


def evaluate(model, episodes, seed, dynamic=True, reward_revision=1, env_class=CityPPOEnv, env_kwargs=None):
    env = env_class(dynamic=dynamic, randomize=True, reward_revision=reward_revision, **(env_kwargs or {}))
    counts = dict(success=0, collision=0, outside=0, timeout=0)
    rewards, minimums, brake_ticks, ticks, safe_successes,projected_ticks = [], [], 0, 0, 0, 0
    for index in range(episodes):
        obs, _ = env.reset(seed=seed + index)
        total, minimum = 0., float('inf')
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            minimum = min(minimum, float(env.scan_history[-1].min()))
            ticks += 1
            brake_ticks += info['direction_braked']
            projected_ticks += int(info.get('component_projected',False))
            if terminated or truncated:
                for key in ('success', 'collision', 'outside'):
                    counts[key] += int(info[key])
                counts['timeout'] += int(truncated)
                break
        rewards.append(total)
        minimums.append(minimum)
        safe_successes += int(info['success'] and minimum >= .8)
    env.close()
    return dict(episodes=episodes, seed=seed, dynamic=dynamic,
                **{key + '_rate': value / episodes for key, value in counts.items()},
                mean_return=float(np.mean(rewards)), minimum_scan_m=min(minimums),
                full_scan_margin_rate=sum(v >= .8 for v in minimums) / episodes,
                safe_success_rate=safe_successes / episodes,
                brake_fraction=brake_ticks / max(1, ticks),projection_fraction=projected_ticks/max(1,ticks))


def city_checkpoint_score(result):
    geometry_safe = result['collision_rate'] <= .01 and result['outside_rate'] == 0
    feasible = geometry_safe and result['safe_success_rate'] >= .95
    return (feasible, geometry_safe, result['safe_success_rate'],
            -result['collision_rate'], -result['outside_rate'], result['success_rate'],
            result['full_scan_margin_rate'], result['mean_return'])


class Checkpoint(BaseCallback):
    def __init__(self, output, dynamic, episodes, reward_revision=1, env_class=CityPPOEnv, env_kwargs=None):
        super().__init__()
        self.output, self.dynamic, self.episodes = output, dynamic, episodes
        self.last = 0
        self.best = None
        self.records = []
        self.started = time.monotonic()
        self.reward_revision = reward_revision
        self.env_class,self.env_kwargs=env_class,env_kwargs

    def check(self):
        result = evaluate(self.model, self.episodes, 3100000, self.dynamic, self.reward_revision,self.env_class,self.env_kwargs)
        score = city_checkpoint_score(result)
        if self.best is None or score > self.best:
            self.best = score
            self.model.save(self.output / 'best_model')
        self.records.append(dict(timesteps=self.num_timesteps,
            elapsed_s=time.monotonic() - self.started, **result))
        (self.output / 'validation.json').write_text(json.dumps(self.records, indent=2) + '\n')
        print(json.dumps(self.records[-1]), flush=True)

    def _on_training_start(self):
        self.check()

    def _on_step(self):
        if self.num_timesteps - self.last >= 25000:
            self.last = self.num_timesteps
            self.check()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--validation-episodes', type=int, default=12)
    parser.add_argument('--eval-episodes', type=int, default=40)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--entropy-coef', type=float, default=.015)
    parser.add_argument('--static-probability', type=float, default=0.)
    parser.add_argument('--reward-revision', type=int, choices=(1, 2), default=1)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--delivery-plan',type=Path)
    parser.add_argument('--obstacle-manifest',type=Path)
    parser.add_argument('--policy-only-training',action='store_true')
    parser.add_argument('--departure-probability',type=float,default=0.)
    parser.add_argument('--replay-probability',type=float,default=0.)
    args = parser.parse_args()
    env_class,extra=CityPPOEnv,{}
    if args.delivery_plan:
        if not args.obstacle_manifest:parser.error('--delivery-plan requires --obstacle-manifest')
        from .delivery_ppo_env import DeliveryPPOEnv
        env_class=DeliveryPPOEnv
        extra=dict(plan_path=args.delivery_plan,manifest_path=args.obstacle_manifest)
    train_extra=dict(extra)
    if args.delivery_plan:
        train_extra.update(policy_only=args.policy_only_training,departure_probability=args.departure_probability,
                           replay_probability=args.replay_probability)
    if min(args.timesteps, args.validation_episodes, args.eval_episodes) < 1:
        parser.error('steps and episode counts must be positive')
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'run_config.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    shutil.copy2(Path(__file__).with_name('city_ppo_env.py'), args.output_dir / 'environment_snapshot.py')
    if args.delivery_plan:
        shutil.copy2(Path(__file__).with_name('delivery_ppo_env.py'),args.output_dir/'delivery_environment_snapshot.py')
    source = PPO.load(args.initial_model, device='cpu')
    if source.observation_space.shape != (153,) or source.action_space.shape != (3,):
        raise ValueError('Expected direct 3D PPO model')
    if not args.evaluate_only:
        probe = env_class(dynamic=not args.static, reward_revision=args.reward_revision,**extra)
        check_env(probe)
        probe.close()
        env = make_vec_env(env_class, n_envs=8, seed=args.seed,
                           env_kwargs=dict(dynamic=not args.static, reward_revision=args.reward_revision,
                                           static_probability=args.static_probability,**train_extra),
                           monitor_dir=str(args.output_dir / 'monitor'))
        model = PPO(source.policy_class, env, learning_rate=args.learning_rate,
                    n_steps=256, batch_size=256, n_epochs=8, gamma=.995,
                    gae_lambda=.95, ent_coef=args.entropy_coef, policy_kwargs=source.policy_kwargs,
                    seed=args.seed, device='cpu', verbose=0,
                    tensorboard_log=str(args.output_dir / 'tensorboard'))
        transfer_policy_weights(source.policy, model.policy)
        model.velocity_mapping = 'radial'
        model.lidar_profile = 'gazebo'
        model.training_scene = 'logistics_city_ppo'
        model.altitude_min=2.
        model.altitude_max=32. if args.delivery_plan else 10.
        callback = Checkpoint(args.output_dir, not args.static, args.validation_episodes, args.reward_revision,env_class,extra)
        model.learn(args.timesteps, callback=callback)
        model.save(args.output_dir / 'final_model')
        env.close()
        source = PPO.load(args.output_dir / 'best_model.zip', device='cpu')
    metrics = evaluate(source, args.eval_episodes, 3200000, not args.static, args.reward_revision,env_class,extra)
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print('HOLDOUT ' + json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
