# Results: formal

This report is generated from the JSONL run records. It does not treat a lower variable count or L value as an application speedup.

## Run scope

- Workload: 240 tasks across 6 circuit families.
- Recorded environment: Python 3.12.14, arm64, 10 logical CPUs; experiment workers use one numerical thread.
- Timing includes task JSON parsing, feature extraction or preprocessing, path planning, and evaluation inside each worker. Shared dependency import and process startup are recorded separately.
- Every per-run total is reconciled as the sum of its top-level recorded stages plus other_s = total_s minus the sum of timings_s. The residual contains work outside the detailed timers, including task parsing and dispatch. Nested selector feature and inference timers are included in its residual.
- Across 6768 recorded runs, other_s ranges from 2.7417e-05 to 0.00494396 seconds; 0 residuals are below -1e-09 seconds.
- The preprocessing stage chart includes Other task overhead (incl. parsing/optimizer setup). The separate stage medians need not sum to the median end-to-end time.
- Process peak RSS is taken from OS high-water marks and sampled RSS; the run metadata documents the measurement scope and polling interval.
- The backend dataset reuses matching native Quimb rows from `preprocess.jsonl` when the source field marks them; record counts therefore are not counts of independent executions. Online selector replays are timed separately.
- The amplitude reference and per-run numerical verification are recorded separately. Failed, timed-out, unsupported, and unverified rows remain in the raw files.

## Measured records

There are 6768 experiment records; 5970 successful records across preprocessing, backend and selector runs carry a passing reference check.

| Experiment | Family / method | Runs | Success | Verified | Median time (s) | Median peak RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|
| preprocess | arithmetic / adaptive | 120 | 120 | 120 | 0.0850682 | 0.063 |
| preprocess | arithmetic / fixed | 120 | 120 | 120 | 0.00719673 | 0.043 |
| preprocess | arithmetic / native | 120 | 120 | 120 | 0.0697409 | 0.062 |
| preprocess | brickwork1d / adaptive | 120 | 120 | 120 | 0.051476 | 0.061 |
| preprocess | brickwork1d / fixed | 120 | 120 | 120 | 0.0134 | 0.046 |
| preprocess | brickwork1d / native | 120 | 120 | 120 | 0.0463152 | 0.060 |
| preprocess | brickwork2d / adaptive | 120 | 120 | 120 | 0.0452532 | 0.061 |
| preprocess | brickwork2d / fixed | 120 | 120 | 120 | 0.0114652 | 0.045 |
| preprocess | brickwork2d / native | 120 | 120 | 120 | 0.0425532 | 0.060 |
| preprocess | graphstate / adaptive | 120 | 120 | 120 | 0.0461986 | 0.061 |
| preprocess | graphstate / fixed | 120 | 120 | 120 | 0.0129676 | 0.046 |
| preprocess | graphstate / native | 120 | 120 | 120 | 0.043621 | 0.060 |
| preprocess | grover / adaptive | 120 | 120 | 90 | 3.84522 | 0.148 |
| preprocess | grover / fixed | 120 | 120 | 90 | 0.15772 | 0.056 |
| preprocess | grover / native | 120 | 120 | 90 | 0.286802 | 0.065 |
| preprocess | iqp / adaptive | 120 | 120 | 120 | 0.423991 | 0.064 |
| preprocess | iqp / fixed | 120 | 120 | 120 | 0.107657 | 0.054 |
| preprocess | iqp / native | 120 | 120 | 120 | 0.0586809 | 0.063 |
| backends | arithmetic / mitm | 120 | 30 | 30 | 0.235358 | 0.231 |
| backends | arithmetic / quimb | 120 | 120 | 120 | 0.0697409 | 0.062 |
| backends | arithmetic / statevector | 120 | 120 | 120 | 0.00753404 | 0.023 |
| backends | brickwork1d / mitm | 120 | 120 | 120 | 0.00189804 | 0.020 |
| backends | brickwork1d / quimb | 120 | 120 | 120 | 0.0463152 | 0.060 |
| backends | brickwork1d / statevector | 120 | 120 | 120 | 0.00160404 | 0.022 |
| backends | brickwork2d / mitm | 120 | 120 | 120 | 0.00216302 | 0.020 |
| backends | brickwork2d / quimb | 120 | 120 | 120 | 0.0425532 | 0.060 |
| backends | brickwork2d / statevector | 120 | 120 | 120 | 0.00174208 | 0.022 |
| backends | graphstate / mitm | 120 | 120 | 120 | 0.00187646 | 0.020 |
| backends | graphstate / quimb | 120 | 120 | 120 | 0.043621 | 0.060 |
| backends | graphstate / statevector | 120 | 90 | 90 | 0.00214871 | 0.023 |
| backends | grover / mitm | 120 | 30 | 30 | 0.351168 | 0.231 |
| backends | grover / quimb | 120 | 120 | 90 | 0.286802 | 0.065 |
| backends | grover / statevector | 120 | 120 | 90 | 0.00747912 | 0.023 |
| backends | iqp / mitm | 120 | 120 | 120 | 0.00203194 | 0.021 |
| backends | iqp / quimb | 120 | 120 | 120 | 0.0586809 | 0.063 |
| backends | iqp / statevector | 120 | 120 | 120 | 0.0016559 | 0.022 |
| selector | arithmetic / effective | 120 | 90 | 90 | 0.00522346 | 0.024 |
| selector | arithmetic / fixed | 120 | 120 | 120 | 0.0850857 | 0.062 |
| selector | arithmetic / raw | 120 | 120 | 120 | 0.00853367 | 0.024 |
| selector | brickwork1d / effective | 120 | 120 | 120 | 0.00215281 | 0.024 |
| selector | brickwork1d / fixed | 120 | 120 | 120 | 0.0531617 | 0.060 |
| selector | brickwork1d / raw | 120 | 120 | 120 | 0.00186862 | 0.023 |
| selector | brickwork2d / effective | 120 | 120 | 120 | 0.00217646 | 0.021 |
| selector | brickwork2d / fixed | 120 | 120 | 120 | 0.051171 | 0.060 |
| selector | brickwork2d / raw | 120 | 120 | 120 | 0.00203935 | 0.021 |
| selector | graphstate / effective | 120 | 120 | 120 | 0.00225881 | 0.021 |
| selector | graphstate / fixed | 120 | 90 | 90 | 0.00218365 | 0.023 |
| selector | graphstate / raw | 120 | 120 | 120 | 0.00200117 | 0.021 |
| selector | grover / effective | 120 | 120 | 90 | 0.0103843 | 0.024 |
| selector | grover / fixed | 120 | 120 | 90 | 0.323638 | 0.065 |
| selector | grover / raw | 120 | 120 | 90 | 0.00862998 | 0.024 |
| selector | iqp / effective | 120 | 120 | 120 | 0.00217598 | 0.023 |
| selector | iqp / fixed | 120 | 120 | 120 | 0.0727668 | 0.064 |
| selector | iqp / raw | 120 | 120 | 120 | 0.00194944 | 0.023 |
| batch | graphstate / independent / 1 | 12 | 12 | 12 | 0.00211827 | 0.020 |
| batch | graphstate / independent / 128 | 12 | 12 | 12 | 0.0401339 | 0.020 |
| batch | graphstate / independent / 32 | 12 | 12 | 12 | 0.0128037 | 0.020 |
| batch | graphstate / independent / 8 | 12 | 12 | 12 | 0.00462883 | 0.020 |
| batch | graphstate / quimb_cached / 1 | 12 | 12 | 12 | 0.0475731 | 0.060 |
| batch | graphstate / quimb_cached / 128 | 12 | 12 | 12 | 0.919035 | 0.093 |
| batch | graphstate / quimb_cached / 32 | 12 | 12 | 12 | 0.292446 | 0.061 |
| batch | graphstate / quimb_cached / 8 | 12 | 12 | 12 | 0.103569 | 0.061 |
| batch | graphstate / shared_graph / 1 | 12 | 12 | 12 | 0.00209367 | 0.020 |
| batch | graphstate / shared_graph / 128 | 12 | 12 | 12 | 0.00553658 | 0.020 |
| batch | graphstate / shared_graph / 32 | 12 | 12 | 12 | 0.00307679 | 0.020 |
| batch | graphstate / shared_graph / 8 | 12 | 12 | 12 | 0.00236471 | 0.020 |
| batch | iqp / independent / 1 | 12 | 12 | 12 | 0.00246519 | 0.021 |
| batch | iqp / independent / 128 | 12 | 12 | 12 | 0.0580126 | 0.021 |
| batch | iqp / independent / 32 | 12 | 12 | 12 | 0.0161909 | 0.021 |
| batch | iqp / independent / 8 | 12 | 12 | 12 | 0.00625413 | 0.021 |
| batch | iqp / quimb_cached / 1 | 12 | 12 | 12 | 0.0873322 | 0.063 |
| batch | iqp / quimb_cached / 128 | 12 | 12 | 12 | 1.499 | 0.095 |
| batch | iqp / quimb_cached / 32 | 12 | 12 | 12 | 0.42698 | 0.064 |
| batch | iqp / quimb_cached / 8 | 12 | 12 | 12 | 0.154553 | 0.063 |
| batch | iqp / shared_graph / 1 | 12 | 12 | 12 | 0.00244335 | 0.021 |
| batch | iqp / shared_graph / 128 | 12 | 12 | 12 | 0.0358799 | 0.021 |
| batch | iqp / shared_graph / 32 | 12 | 12 | 12 | 0.0101888 | 0.021 |
| batch | iqp / shared_graph / 8 | 12 | 12 | 12 | 0.00498296 | 0.021 |

## Figures

| Figure | PNG | SVG |
|---|---|---|
| Preprocessing | ![Preprocessing](preprocess.png) | [SVG](preprocess.svg) |
| Preprocessing stage costs | ![Preprocessing stage costs](preprocess_stages.png) | [SVG](preprocess_stages.svg) |
| Backend selection | ![Backend selection](selector.png) | [SVG](selector.svg) |
| Batch overview | ![Batch overview](batch.png) | [SVG](batch.svg) |
| Batch cost by circuit | ![Batch cost by circuit](batch_by_task.png) | [SVG](batch_by_task.svg) |

## Paired results

A pair is included only when every repeat for both methods completed and passed reference verification.

- Preprocessing `fixed`: median native/strategy time = 3.288× over 230 tasks; candidate faster on 202 and slower on 28.
- Preprocessing `adaptive`: median native/strategy time = 0.820× over 230 tasks; candidate faster on 0 and slower on 230.
- Selector `raw`: median fixed/raw end-to-end time = 28.004× over 220 tasks.
- Selector `effective`: median fixed/effective end-to-end time = 25.633× over 210 tasks.
- Selector `effective_vs_raw`: median raw/effective end-to-end time = 0.886× over 220 tasks.

## Structure and end-to-end cost

Among 240 tasks with rank-step metrics, rank decreased on 24; L increased on 0 of those. Estimated combo increased on 0 of 24 rank-decrease tasks with comparable candidate estimates. Across all comparable tasks, estimated combo increased on 1 of 240; this includes tasks where rank did not decrease.
The examples below compare paired medians for the complete adaptive and native workflows. Adaptive cost includes candidate generation, planning, evaluation, and task overhead; the smaller rank/L is a structural observation, not evidence that the rank change caused the runtime difference.

| Task | Verified repeats | Rank r | L | Native median (s) | Adaptive median (s) | Adaptive/native |
|---|---:|---:|---:|---:|---:|---:|
| brickwork1d-n8-s108 | 3 | 8 → 6 | 72 → 50 | 0.0482584 | 0.332624 | 6.89× |
| brickwork1d-n8-s109 | 3 | 8 → 7 | 57 → 55 | 0.0466686 | 0.193112 | 4.14× |
| brickwork1d-n10-s108 | 3 | 9 → 8 | 63 → 58 | 0.0482848 | 0.0935176 | 1.94× |

## Offline backend-cost oracle regret

These are out-of-fold model predictions compared with the fastest measured backend per task. They are separate from measured online selector replays above. OOF prediction completion is not an independent amplitude verification.

| Policy | OOF tasks | Predictions completed | Median estimated total (s) | Median oracle regret (×) |
|---|---:|---:|---:|---:|
| effective | 240 | 230 | 0.00255956 | 1.000 |
| fixed | 240 | 230 | 0.0524148 | 28.681 |
| raw | 240 | 240 | 0.00214069 | 1.000 |

Selector routes include feature extraction and inference. The PAR2 view assigns each failed, timed-out, unsupported, or unverified replay twice the configured timeout; it complements the verified-pair comparison.

| Selector | Recorded replays | Correctly completed | Completion rate | Total PAR2 (s) |
|---|---:|---:|---:|---:|
| effective | 720 | 660 | 92% | 7202.43 |
| fixed | 720 | 660 | 92% | 7314.26 |
| raw | 720 | 690 | 96% | 3604.24 |

## Batch reuse by circuit

Each break-even entry is the first tested batch size for which the shared graph run is faster. It is reported only when every repeat for both compared methods passed verification.

| Task | Physical qubits | Unique queries by batch size | First shared win vs independent | First shared win vs Quimb cache |
|---|---:|---|---:|---:|
| graphstate-n10-s100 | 28 | 1: 1, 8: 7, 32: 16, 128: 16 | 1 | 1 |
| graphstate-n4-s100 | 4 | 1: 1, 8: 7, 32: 16, 128: 16 | 8 | 1 |
| graphstate-n6-s100 | 12 | 1: 1, 8: 7, 32: 16, 128: 16 | 1 | 1 |
| graphstate-n8-s100 | 20 | 1: 1, 8: 7, 32: 16, 128: 16 | 1 | 1 |
| iqp-n10-s100 | 10 | 1: 1, 8: 8, 32: 32, 128: 128 | 8 | 1 |
| iqp-n4-s100 | 4 | 1: 1, 8: 8, 32: 32, 128: 119 | 1 | 1 |
| iqp-n6-s100 | 6 | 1: 1, 8: 8, 32: 32, 128: 128 | 8 | 1 |
| iqp-n8-s100 | 8 | 1: 1, 8: 8, 32: 32, 128: 128 | 1 | 1 |

## Reproduction files

The JSONL files preserve individual instances and failures. `summary.json` and the per-experiment CSV files are derived summaries. Re-run the report command after adding run records to rebuild this page and all charts.

This is a small, machine-specific CPU benchmark. A benefit is claimed only for the tested workload, where the measured end-to-end results and correctness checks support it.
