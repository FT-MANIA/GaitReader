"""Run fixed-VQ gait-language SSL ablation experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime

import run


ABLATIONS = (
    "full",
    "no_bilateral_pair",
    "no_rhythm",
    "no_contralateral",
    "cross_dof_soft_only",
    "contralateral_soft_only",
)


def get_args() -> argparse.Namespace:
    """Parse the shared training options plus the requested ablations."""
    parser = run.build_parser()
    parser.prog = "run_exp.py"
    parser.description = "Fixed-VQ gait-language SSL ablation experiments"
    parser.set_defaults(
        stage="ssl_downstream",
        output_dir="Results/gait_language/ablation_exp",
        vq_checkpoint=(
            "Results/gait_language/dev_exp/"
            "dev_exp_0826_1542/best_vq.pt"
        ),
    )
    parser.add_argument(
        "--ablation",
        action="append",
        choices=ABLATIONS,
        help="Ablation to run; repeat the option to select multiple variants",
    )
    return parser.parse_args()


def configure_ablation(
    base_args: argparse.Namespace, name: str
) -> argparse.Namespace:
    """Create one ablation configuration from the complete new task."""
    args = deepcopy(base_args)
    args.stage = "ssl_downstream"
    args.ablation_name = name
    args.run_name = f"ablation_exp_{datetime.now().strftime('%m%d_%H%M')}"

    args.ssl_within_task = True
    args.ssl_cross_dof_task = True
    args.ssl_rhythm_task = True
    args.ssl_bilateral_context_task = False
    args.ssl_contralateral_task = True
    args.ssl_bilateral_pair_task = True
    args.ssl_swap_task = False

    args.cross_dof_hard_code_weight = 0.0
    args.cross_dof_soft_code_weight = 1.0
    args.cross_dof_prototype_weight = 1.0
    args.contralateral_hard_code_weight = 0.0
    args.contralateral_soft_code_weight = 1.0
    args.contralateral_prototype_weight = 1.0

    if name == "no_bilateral_pair":
        args.ssl_bilateral_pair_task = False
    elif name == "no_rhythm":
        args.ssl_rhythm_task = False
    elif name == "no_contralateral":
        args.ssl_contralateral_task = False
    elif name == "cross_dof_soft_only":
        args.cross_dof_prototype_weight = 0.0
    elif name == "contralateral_soft_only":
        args.contralateral_prototype_weight = 0.0

    return args


def main() -> None:
    """Run the selected ablations in their prescribed order."""
    base_args = get_args()
    selected = base_args.ablation or ABLATIONS
    for name in selected:
        args = configure_ablation(base_args, name)
        print(f"\nRunning ablation: {name}")
        run.run_experiment(args)


if __name__ == "__main__":
    main()
