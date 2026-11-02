import os
import shutil
from datetime import datetime

import gymnasium as gym
import torch

from noisy_net import evaluate_policy, str2bool, NoisyNetDQN_agent
from noisy_net.parser import parse_arguments


def main():
    opt = parse_arguments()
    opt.dvc = torch.device(opt.dvc)  # from str to torch.device
    EnvName = ["CartPole-v1", "LunarLander-v2"]
    BriefEnvName = ["CPV1", "LLdV2"]
    env = gym.make(EnvName[opt.EnvIdex], render_mode="human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    algo_name = "NoisyNetDQN"

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed: {opt.seed}")

    print(
        "Algorithm:",
        algo_name,
        "  Env:",
        BriefEnvName[opt.EnvIdex],
        "  state_dim:",
        opt.state_dim,
        "  action_dim:",
        opt.action_dim,
        "  Random Seed:",
        opt.seed,
        "  max_e_steps:",
        opt.max_e_steps,
        "\n",
    )

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter

        timenow = str(datetime.now())[0:-10]
        timenow = " " + timenow[0:13] + "_" + timenow[-2::]
        writepath = (
            f"runs/{algo_name}-{BriefEnvName[opt.EnvIdex]}_S{opt.seed}_" + timenow
        )
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Build model and replay buffer
    if not os.path.exists("model"):
        os.mkdir("model")
    agent = NoisyNetDQN_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            print(
                "EnvName:",
                BriefEnvName[opt.EnvIdex],
                "seed:",
                opt.seed,
                "score:",
                score,
            )
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, _info = env.reset(
                seed=env_seed
            )  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            """Interact & trian"""
            while not done:
                # e-greedy exploration
                if total_steps < opt.random_steps:
                    a = env.action_space.sample()
                else:
                    a = agent.select_action(s)  # NoisyNet无需e-greedy
                s_next, r, dw, tr, _info = env.step(a)  # dw: dead&win; tr: truncated
                done = dw or tr

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next

                """Update"""
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if (
                    total_steps >= opt.random_steps
                    and total_steps % opt.update_every == 0
                ):
                    for j in range(opt.update_every):
                        agent.train()

                """Record & Log"""
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar("ep_r", score, global_step=total_steps)
                    print(
                        "EnvName:",
                        BriefEnvName[opt.EnvIdex],
                        "seed:",
                        opt.seed,
                        f"steps: {int(total_steps / 1000)}k",
                        "score:",
                        int(score),
                    )
                total_steps += 1

                """Save model"""
                if total_steps % opt.save_interval == 0:
                    agent.save(
                        algo_name, BriefEnvName[opt.EnvIdex], int(total_steps / 1000)
                    )
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
