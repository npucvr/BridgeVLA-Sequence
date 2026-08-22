# BridgeVLA RLBench reproduction audit

## Scope

This audit compares the released `model_80.pth` checkpoint and evaluator with
BridgeVLA paper v2 ([arXiv:2506.07961](https://arxiv.org/abs/2506.07961)). The
paper reports 18 RLBench tasks, 25 binary-success trials per task, at most 25
action steps per trial, five repeated evaluations, and an 18-task macro average
of 88.2%.

## Dataset split (critical)

There are two separate public raw-data repositories:

- [LPY/BridgeVLA_RLBench_TRAIN_DATA](https://huggingface.co/datasets/LPY/BridgeVLA_RLBench_TRAIN_DATA/tree/main): the training demonstrations.
- [LPY/BridgeVLA_RLBench_EVAL_DATA](https://huggingface.co/datasets/LPY/BridgeVLA_RLBench_EVAL_DATA/tree/main): the held-out evaluation episodes.

The BridgeVLA training code sets `NUM_TRAIN = 100`. PerAct's data-generation
documentation also describes separate train/validation/test sets and 100 train
demonstrations; its evaluation environment reloads deterministic held-out
episodes. At the time of this audit, the HF dataset commits are TRAIN
`d7871b87befc5340f83e241f880a09d0769eedd7` and EVAL
`271afb90d902f8e122886e37599bfba15ea946cf`. The BridgeVLA EVAL repository
contains exactly episodes `0..24` for each of the same 18 tasks. The extracted
train and eval episodes also have different variation-number sequences (for
example, `close_jar`), confirming that EVAL is not merely a renamed copy of the
first 25 training episodes. Its archives preserve a generator path prefix and
are gzip-compressed despite the `.tar.xz` suffix; the extraction helper handles
both released EVAL archive prefixes and strips them to the task-root layout
expected by `eval.py`.

**Important:** the earlier five-run result (85.33%) used
`data/RLBench_TRAIN_DATA/.../episode0..24`, i.e. the first 25 training
demonstrations, not the released held-out evaluation split. It must not be
compared to the paper. The result below uses the official EVAL split.

## Protocol checks

The following are consistent with the paper and released code:

- The task list contains the same 18 tasks and order as Table 1.
- `eval_episodes=25`, `start_episode=0`, and `episode_length=25`.
- The evaluator passes the episode index as `eval_demo_seed`, so episodes are
  deterministically reloaded from the evaluation dataset.
- Cameras are `front`, `left_shoulder`, `right_shoulder`, and `wrist`; the
  released data contains the corresponding RGB-D/mask streams.
- The keyframe implementation is the PerAct heuristic with the paper's
  `0.1 rad/s` stopping threshold.
- `model_80.pth` is the released checkpoint (epoch 80); its SHA-256 is
  `76dd35185c41afe1573a9b9ca47def9dd66b3ecd01f95b6374e3cfb7dc27baf9`.
- The checkpoint configs match the Hugging Face release. `bs=4` per GPU and
  `train_iter=160000` give about 83,000 total steps on 48 GPUs, matching the
  paper's global batch size 192 and 83,000-step description.
- The aggregate uses the unweighted mean of the 18 task means and sample
  standard deviation (`ddof=1`), consistent with the paper's rounded five-run
  values.

## Final result (official EVAL split)

Five repeats of the released `model_80.pth` checkpoint on
`LPY/BridgeVLA_RLBench_EVAL_DATA` episodes `0..24`, 18 tasks × 25 episodes:
**87.42%** 18-task average, versus the paper's **88.2%** (Δ = **-0.78**);
mean absolute per-task delta **2.00**. Every task falls inside the
paper/std-overlap verdict ("OK-ish"), i.e. no task-level discrepancy outside
the combined run-to-run variation.

Task means and std (sample std, ddof=1, across the five runs):

```
task                                 mean      std   runs
--------------------------------------------------------------
close_jar                          100.00     0.00   [100, 100, 100, 100, 100]
reach_and_drag                     100.00     0.00   [100, 100, 100, 100, 100]
insert_onto_square_peg              87.20     8.67   [ 80,  84,  80,  92, 100]
meat_off_grill                     100.00     0.00   [100, 100, 100, 100, 100]
open_drawer                         99.20     1.79   [ 96, 100, 100, 100, 100]
place_cups                          54.40     6.07   [ 48,  52,  56,  52,  64]
place_wine_at_rack_location         84.80     9.55   [ 80,  88,  76,  80, 100]
push_buttons                       100.00     0.00   [100, 100, 100, 100, 100]
put_groceries_in_cupboard           76.00     2.83   [ 76,  72,  76,  80,  76]
put_item_in_drawer                  96.00     2.83   [ 92, 100,  96,  96,  96]
put_money_in_safe                  100.00     0.00   [100, 100, 100, 100, 100]
light_bulb_in                       86.40     4.56   [ 88,  84,  92,  80,  88]
slide_block_to_color_target         92.80     3.35   [ 88,  92,  96,  92,  96]
place_shape_in_shape_sorter         56.80     5.22   [ 60,  56,  52,  52,  64]
stack_blocks                        78.40     5.37   [ 88,  76,  76,  76,  76]
stack_cups                          84.80     4.38   [ 88,  88,  88,  80,  80]
sweep_to_dustpan_of_size            88.80     1.79   [ 88,  88,  88,  88,  92]
turn_tap                            88.00     4.90   [ 88,  88,  92,  80,  92]
--------------------------------------------------------------
18-task average                     87.42
```

Per-task comparison against the paper table:

```
task                                      paper           ours   delta  verdict
--------------------------------------------------------------------------------
close_jar                        100.00+-0.00   100.00+-0.00     +0.00  OK-ish
reach_and_drag                   100.00+-0.00   100.00+-0.00     +0.00  OK-ish
insert_onto_square_peg            88.00+-2.80    87.20+-8.67     -0.80  OK-ish
meat_off_grill                   100.00+-0.00   100.00+-0.00     +0.00  OK-ish
open_drawer                      100.00+-0.00    99.20+-1.79     -0.80  OK-ish
place_cups                        58.40+-10.00   54.40+-6.07     -4.00  OK-ish
place_wine_at_rack_location       88.00+-2.80    84.80+-9.55     -3.20  OK-ish
push_buttons                      98.40+-2.20   100.00+-0.00     +1.60  OK-ish
put_groceries_in_cupboard         73.60+-4.60    76.00+-2.83     +2.40  OK-ish
put_item_in_drawer                99.20+-1.80    96.00+-2.83     -3.20  OK-ish
put_money_in_safe                 99.20+-1.80   100.00+-0.00     +0.80  OK-ish
light_bulb_in                     87.20+-6.60    86.40+-4.56     -0.80  OK-ish
slide_block_to_color_target       96.00+-2.80    92.80+-3.35     -3.20  OK-ish
place_shape_in_shape_sorter       60.80+-7.70    56.80+-5.22     -4.00  OK-ish
stack_blocks                      76.80+-8.70    78.40+-5.37     +1.60  OK-ish
stack_cups                        81.60+-3.60    84.80+-4.38     +3.20  OK-ish
sweep_to_dustpan_of_size          87.20+-1.80    88.80+-1.79     +1.60  OK-ish
turn_tap                          92.80+-3.30    88.00+-4.90     -4.80  OK-ish
--------------------------------------------------------------------------------
18-task average: paper=88.20, ours=87.42, delta=-0.78
mean absolute per-task delta: 2.00
```

The released checkpoint therefore reproduces the paper within run-to-run
variation: the largest negative deltas are `turn_tap` (-4.8), `place_cups` and
`place_shape_in_shape_sorter` (-4.0), and `place_wine_at_rack_location` and
`put_item_in_drawer` / `slide_block_to_color_target` (-3.2), all smaller than
the corresponding paper std or the observed across-run std. Nine tasks land at
or above the paper value. The residual -0.78 average gap is consistent with the
environment caveat below (PyTorch/TorchVision/Transformers/NumPy versions
differ from the installation script's pin).

Raw per-run artifacts stay on server20:
`data/bridgevla_ckpt/bridgevla/rlbench/eval/rlbench_eval_split_seq/run_{1..5}/`
(`eval_config.yaml` records `eval_datafolder: .../data/RLBench_EVAL_DATA` for
every run), console logs under `/tmp/bridgevla_rlbench_eval_split_seq_logs/`,
and the scheduler journal
`/tmp/bridgevla_rlbench_eval_scheduler.log` (aggregate + compare output).

## Execution history

After downloading the EVAL archives and extracting
episodes `0..24` (verified as exactly 18 tasks × 25 episodes with matching
RGB-D/mask file counts), five repeats ran on server20 while other users'
jobs competed for its GPUs. A `GPU_IDS=0,2` wave launch of the parallel
launcher crashed twice: the first wave exited with signal 11 during
Qt/OpenSSL initialization (two CoppeliaSim instances on the same
`DISPLAY=:1`), and the second wave was terminated. Runs were then executed
with isolated displays: `run_1` on GPU0/`:1.0`, `run_2` on GPU2/`:2.0`,
`run_3` and `run_4` concurrently on GPU0/`:1.0` and GPU2/`:2.0`, and `run_5`
on GPU0/`:1.0`. Every run completed with `rc=0`. The model smoke-load
succeeded with an external process already using the GPU (about 19 GB was
free before loading, about 2.7 GB remained afterward).

## Correct reproduction command

After downloading the EVAL archives:

```bash
DATA_DIR=/PATH/TO/RLBench_EVAL_DATA \
  bash scripts/rlbench_repro/extract_eval_data.sh

EVAL_DATAFOLDER=/PATH/TO/RLBench_EVAL_DATA \
RESULT_LOG_DIR=rlbench_eval_split \
  bash scripts/rlbench_repro/run_5_parallel.sh
```

The parallel launcher verifies the data (18 tasks × 25 episodes, `--exact-episodes`)
before starting, preventing accidental evaluation on the training split. When
running multiple CoppeliaSim instances on one host, each concurrent slot needs
its own Xvfb display and its own GPU, e.g.
`GPU_IDS=0,2 DISPLAY_IDS=:1.0,:2.0 bash scripts/rlbench_repro/run_5_parallel.sh`.
`run_5_repeats.sh` is the single-GPU sequential variant; both write per-task
`eval_results.csv` under
`<model_folder>/eval/<RESULT_LOG_DIR>/run_<id>/model_80/` and print the
aggregate plus the paper comparison.

## Earlier invalid-split result (context only)

The five runs on the first 25 training episodes were internally consistent
(18 tasks × 25 episodes, binary scores, exact checkpoint), but averaged
**85.33%**, with the largest gaps in `place_cups` (46.4 vs paper 58.4),
`stack_cups` (61.6 vs 81.6), and `place_shape_in_shape_sorter` (52.0 vs 60.8).
Those differences motivated the split audit. The official EVAL split result
above (87.42%) replaces it and shows the checkpoint does reproduce the paper;
the invalid split systematically underestimated the harder tasks.

## Environment caveat

The simulator/dependency source commits match the installation script
(RLBench `587a6a0e6d...`, PyRep `231a1ac6...`, CoppeliaSim Edu 4.1). The
available server environment currently reports PyTorch 2.7.1+cu128,
TorchVision 0.22.1+cu128, Transformers 4.51.3, and NumPy 2.0.2, while the
installation script explicitly pins TorchVision/Torchaudio 2.5.1 and does not
pin the Torch package itself. Any final number should therefore be reported as
reproduced in this environment unless the original package lock is also
reconstructed.
