from argparse import Namespace
import argparse

from noisy_net import str2bool

def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dvc", type=str, default="cpu", help="running device: cuda or cpu"
    )
    parser.add_argument("--EnvIdex", type=int, default=0, help="CP-v1, LLd-v2")
    parser.add_argument(
        "--write",
        type=str2bool,
        default=False,
        help="Use SummaryWriter to record the training",
    )
    parser.add_argument("--render", type=str2bool, default=False, help="Render or Not")
    parser.add_argument(
        "--Loadmodel", type=str2bool, default=False, help="Load pretrained model or Not"
    )
    parser.add_argument("--ModelIdex", type=int, default=100, help="which model to load")

    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--Max_train_steps", type=int, default=int(6e5), help="Max training steps"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=int(50e3),
        help="Model saving interval, in steps.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=int(5e3),
        help="Model evaluating interval, in steps.",
    )
    parser.add_argument(
        "--random_steps",
        type=int,
        default=int(3e3),
        help="steps for random policy to explore",
    )
    parser.add_argument("--update_every", type=int, default=50, help="training frequency")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discounted Factor")
    parser.add_argument("--net_width", type=int, default=200, help="Hidden net width")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="lenth of sliced trajectory"
    )
    opt = parser.parse_args()
    opt.buffer_size = min(int(1e6), opt.Max_train_steps)
    return opt
