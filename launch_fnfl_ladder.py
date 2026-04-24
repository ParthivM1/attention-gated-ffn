#!/usr/bin/env python
"""Launch the Flow-Native FFN (FNFL) experimental ladder on Modal.

Budget: ~$12 = ~3-4 hours A100 time = ~30-50 locked smoke runs.
Locked smoke cost: ~$0.40 per run (5 min on A100 + 30% overhead).

Experimental ladder:
  Step 0: Local CPU sanity (done)
  Step 1: Plain ViT x3 seeds (baseline)
  Step 2: GeoViT residual x3 seeds (baseline)
  Step 3: FNFL k2 t2 x3 seeds (main hypothesis)
  Step 4: FNFL k2 t4 x2 seeds (if step 3 positive, ablate depth)
  Step 5: FNFL k2 t2 noskip x2 seeds (if step 3 close, ablate annealing)
  Step 6: Best variant at 10k/2k scale x1 seed (if step 3 wins decisively)

Decision gates:
  After step 3: FNFL >= plain + 1.5 absolute → continue to step 4-6
              : FNFL within ±0.5 of plain → do step 5 ablation
              : FNFL < plain - 1.0 → STOP
"""

import os
import subprocess
import sys
from pathlib import Path


def launch_modal_run(preset_name: str, seed: int = None, run_suffix: str = "") -> str:
    """Launch a single Modal training run.

    Returns: run name (for monitoring)
    """
    cmd = ["modal", "run", "modal_geovit.py"]
    env = os.environ.copy()
    env["GEOVIT_PROPER_PRESET"] = preset_name

    if seed is not None:
        seed_str = f"_seed{seed}"
        env["GEOVIT_PROPER_TRAIN_ARGS"] = f"--seed {seed}"
    else:
        seed_str = ""

    run_name = f"{preset_name}{seed_str}{run_suffix}"
    print(f"\n{'='*70}")
    print(f"Launching: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Preset: {preset_name}")
    if seed is not None:
        print(f"Seed: {seed}")
    print('='*70)

    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"✓ Launched {run_name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to launch {run_name}: {e}")
        sys.exit(1)

    return run_name


def main():
    print("\n" + "="*70)
    print("FNFL Experimental Ladder Launcher")
    print("="*70)
    print(__doc__)

    # Step 0: Local sanity (done)
    print("\n[STEP 0] Local CPU sanity check")
    print("Status: COMPLETED (run test_fd_transformer_local.py)")

    # Parse command-line args to see which steps to run
    if len(sys.argv) > 1:
        steps_to_run = [int(s) for s in sys.argv[1].split(",")]
    else:
        print("\nUsage: python launch_fnfl_ladder.py <steps>")
        print("  Example: python launch_fnfl_ladder.py 1,2,3")
        print("  Or: python launch_fnfl_ladder.py 3  (just step 3)")
        print("\nRunning all steps 1-3 by default...")
        steps_to_run = [1, 2, 3]

    # Step 1: Plain ViT baseline
    if 1 in steps_to_run:
        print("\n[STEP 1] Plain ViT baseline (x3 seeds)")
        print("Preset: smoke_plain_locked_mid192d6_e40")
        print("Cost estimate: $1.20 (3 runs × $0.40/run)")
        input("Press Enter to continue, Ctrl+C to cancel...")
        for seed in [0, 1, 2]:
            launch_modal_run("smoke_plain_locked_mid192d6_e40", seed=seed)

    # Step 2: GeoViT residual baseline
    if 2 in steps_to_run:
        print("\n[STEP 2] GeoViT residual baseline (x3 seeds)")
        print("Preset: smoke_geovit_simple_locked_mid192d6_e40")
        print("Cost estimate: $1.20 (3 runs × $0.40/run)")
        input("Press Enter to continue, Ctrl+C to cancel...")
        for seed in [0, 1, 2]:
            launch_modal_run("smoke_geovit_simple_locked_mid192d6_e40", seed=seed)

    # Step 3: FNFL k2 t2 (main hypothesis)
    if 3 in steps_to_run:
        print("\n[STEP 3] FNFL k2 t2 (x3 seeds) — MAIN COMPARISON")
        print("Preset: smoke_fnfl_k2_t2_locked_mid192d6_e40")
        print("Cost estimate: $1.20 (3 runs × $0.40/run)")
        print("\nThis is the critical comparison. FNFL must beat plain ViT by >= +1.5 absolute")
        print("to justify continuing to deeper ablations and scaling.")
        input("Press Enter to continue, Ctrl+C to cancel...")
        for seed in [0, 1, 2]:
            launch_modal_run("smoke_fnfl_k2_t2_locked_mid192d6_e40", seed=seed)

        print("\n" + "="*70)
        print("DECISION POINT AFTER STEP 3")
        print("="*70)
        print("Once results are in, compare:")
        print("  Mean(FNFL) vs Mean(plain) — need >= +1.5 absolute")
        print("  Std(FNFL) vs Std(plain) — bands should not overlap much")
        print("  If FNFL wins: continue to steps 4-6")
        print("  If FNFL within ±0.5: run step 5 (ablate annealing)")
        print("  If FNFL loses by > 1: STOP")

    # Step 4: Deeper flow (optional)
    if 4 in steps_to_run:
        print("\n[STEP 4] FNFL k2 t4 (deeper flow, x2 seeds)")
        print("Preset: smoke_fnfl_k2_t4_locked_mid192d6_e40")
        print("Cost estimate: $0.80 (2 runs × $0.40/run)")
        print("Only run if step 3 showed FNFL >> plain ViT")
        input("Press Enter to continue, Ctrl+C to cancel...")
        for seed in [0, 1]:
            launch_modal_run("smoke_fnfl_k2_t4_locked_mid192d6_e40", seed=seed)

    # Step 5: No-skip ablation (optional)
    if 5 in steps_to_run:
        print("\n[STEP 5] FNFL k2 t2 noskip ablation (x2 seeds)")
        print("Preset: smoke_fnfl_k2_t2_noskip_locked_mid192d6_e40")
        print("Cost estimate: $0.80 (2 runs × $0.40/run)")
        print("Run if step 3 FNFL was close to plain (within ±0.5)")
        print("Tests whether the annealing curriculum is needed")
        input("Press Enter to continue, Ctrl+C to cancel...")
        for seed in [0, 1]:
            launch_modal_run("smoke_fnfl_k2_t2_noskip_locked_mid192d6_e40", seed=seed)

    # Step 6: Medium-scale (optional, requires separate preset)
    if 6 in steps_to_run:
        print("\n[STEP 6] Best variant at medium scale (10k/2k, e20, x1 seed)")
        print("Cost estimate: $2.50")
        print("Only run if FNFL clearly beats baselines on smoke ladder")
        print("\nNote: Need to create medium-scale preset first")
        input("Press Enter to continue, Ctrl+C to cancel...")
        # TODO: implement after identifying best variant from steps 1-5


if __name__ == "__main__":
    main()
