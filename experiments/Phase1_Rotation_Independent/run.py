"""
Phase 1 R2: Rotation Independent Training Baseline

Wrapper that calls the shared independent training script with rotation config.
"""

from experiments.Phase1_Scale_Independent.run import run_experiment, main

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0]] + [a if a != "--transform" else "--transform" for a in sys.argv[1:]]
    if "--transform" not in sys.argv:
        sys.argv.extend(["--transform", "rotation"])
    main()
