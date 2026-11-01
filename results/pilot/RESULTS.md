# Results: pilot

This report is generated from the JSONL run records. It does not treat a lower variable count or L value as an application speedup.

## Run scope

- Workload: 48 tasks across 4 circuit families.
- Recorded environment: Python 3.12.14, arm64, 10 logical CPUs; experiment workers use one numerical thread.
- Timing includes task JSON parsing, feature extraction or preprocessing, path planning, and evaluation inside each worker. Shared dependency import and process startup are recorded separately.
- Every per-run total is reconciled as the sum of its top-level recorded stages plus other_s = total_s minus the sum of timings_s. The residual contains work outside the detailed timers, including task parsing and dispatch. Nested selector feature and inference timers are included in its residual.
- Across 1584 recorded runs, other_s ranges from 3.0334e-05 to 0.00370183 seconds; 0 residuals are below -1e-09 seconds.
- The preprocessing stage chart includes Other task overhead (incl. parsing/optimizer setup). The separate stage medians need not sum to the median end-to-end time.
- Process peak RSS is taken from OS high-water marks and sampled RSS; the run metadata documents the measurement scope and polling interval.
- The backend dataset reuses matching native Quimb rows from `preprocess.jsonl` when the source field marks them; record counts therefore are not counts of independent executions. Online selector replays are timed separately.
- The amplitude reference and per-run numerical verification are recorded separately. Failed, timed-out, unsupported, and unverified rows remain in the raw files.

## Measured records

There are 1584 experiment records; 1125 successful records across preprocessing, backend and selector runs carry a passing reference check.

| Experiment | Family / method | Runs | Success | Verified | Median time (s) | Median peak RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|
| preprocess | arithmetic / adaptive | 36 | 36 | 36 | 0.0827387 | 0.062 |
| preprocess | arithmetic / fixed | 36 | 36 | 36 | 0.00714363 | 0.043 |
| preprocess | arithmetic / native | 36 | 36 | 36 | 0.072165 | 0.061 |
| preprocess | graphstate / adaptive | 36 | 36 | 36 | 0.0461891 | 0.061 |
| preprocess | graphstate / fixed | 36 | 36 | 36 | 0.0130178 | 0.046 |
| preprocess | graphstate / native | 36 | 36 | 36 | 0.0429941 | 0.060 |
| preprocess | grover / adaptive | 36 | 36 | 27 | 3.58923 | 0.152 |
| preprocess | grover / fixed | 36 | 36 | 27 | 0.161071 | 0.055 |
| preprocess | grover / native | 36 | 36 | 27 | 0.292469 | 0.065 |
| preprocess | iqp / adaptive | 36 | 36 | 36 | 0.406837 | 0.064 |
| preprocess | iqp / fixed | 36 | 36 | 36 | 0.105137 | 0.054 |
| preprocess | iqp / native | 36 | 36 | 36 | 0.113996 | 0.064 |
| backends | arithmetic / mitm | 36 | 9 | 9 | 0.236781 | 0.230 |
| backends | arithmetic / quimb | 36 | 36 | 36 | 0.072165 | 0.061 |
| backends | arithmetic / statevector | 36 | 36 | 36 | 0.00708523 | 0.023 |
| backends | graphstate / mitm | 36 | 36 | 36 | 0.0018344 | 0.020 |
| backends | graphstate / quimb | 36 | 36 | 36 | 0.0429941 | 0.060 |
| backends | graphstate / statevector | 36 | 27 | 27 | 0.00207783 | 0.023 |
| backends | grover / mitm | 36 | 9 | 9 | 0.354675 | 0.230 |
| backends | grover / quimb | 36 | 36 | 27 | 0.292469 | 0.065 |
| backends | grover / statevector | 36 | 36 | 27 | 0.00749406 | 0.023 |
| backends | iqp / mitm | 36 | 36 | 36 | 0.00192667 | 0.020 |
| backends | iqp / quimb | 36 | 36 | 36 | 0.113996 | 0.064 |
| backends | iqp / statevector | 36 | 36 | 36 | 0.00165877 | 0.022 |
| selector | arithmetic / effective | 36 | 9 | 9 | 0.236012 | 0.231 |
| selector | arithmetic / fixed | 36 | 36 | 36 | 0.068898 | 0.061 |
| selector | arithmetic / raw | 36 | 36 | 36 | 0.00727348 | 0.024 |
| selector | graphstate / effective | 36 | 36 | 36 | 0.0021396 | 0.021 |
| selector | graphstate / fixed | 36 | 27 | 27 | 0.00224975 | 0.023 |
| selector | graphstate / raw | 36 | 36 | 36 | 0.00198754 | 0.020 |
| selector | grover / effective | 36 | 36 | 27 | 0.00907785 | 0.024 |
| selector | grover / fixed | 36 | 36 | 27 | 0.284269 | 0.065 |
| selector | grover / raw | 36 | 36 | 27 | 0.169578 | 0.045 |
| selector | iqp / effective | 36 | 36 | 36 | 0.00200781 | 0.024 |
| selector | iqp / fixed | 36 | 36 | 36 | 0.0897687 | 0.063 |
| selector | iqp / raw | 36 | 36 | 36 | 0.00177165 | 0.023 |
| batch | graphstate / independent / 1 | 12 | 12 | 12 | 0.00192167 | 0.020 |
| batch | graphstate / independent / 128 | 12 | 12 | 12 | 0.035695 | 0.020 |
| batch | graphstate / independent / 32 | 12 | 12 | 12 | 0.0101103 | 0.020 |
| batch | graphstate / independent / 8 | 12 | 12 | 12 | 0.0037856 | 0.020 |
| batch | graphstate / quimb_cached / 1 | 12 | 12 | 12 | 0.0422536 | 0.059 |
| batch | graphstate / quimb_cached / 128 | 12 | 12 | 12 | 0.803791 | 0.092 |
| batch | graphstate / quimb_cached / 32 | 12 | 12 | 12 | 0.228158 | 0.061 |
| batch | graphstate / quimb_cached / 8 | 12 | 12 | 12 | 0.0856095 | 0.061 |
| batch | graphstate / shared_graph / 1 | 12 | 12 | 12 | 0.00183283 | 0.020 |
| batch | graphstate / shared_graph / 128 | 12 | 12 | 12 | 0.0046826 | 0.020 |
| batch | graphstate / shared_graph / 32 | 12 | 12 | 12 | 0.0026721 | 0.020 |
| batch | graphstate / shared_graph / 8 | 12 | 12 | 12 | 0.00204046 | 0.020 |
| batch | iqp / independent / 1 | 12 | 12 | 12 | 0.00205738 | 0.020 |
| batch | iqp / independent / 128 | 12 | 12 | 12 | 0.0532578 | 0.020 |
| batch | iqp / independent / 32 | 12 | 12 | 12 | 0.0144959 | 0.020 |
| batch | iqp / independent / 8 | 12 | 12 | 12 | 0.00486019 | 0.020 |
| batch | iqp / quimb_cached / 1 | 12 | 12 | 12 | 0.0905721 | 0.063 |
| batch | iqp / quimb_cached / 128 | 12 | 12 | 12 | 1.21831 | 0.095 |
| batch | iqp / quimb_cached / 32 | 12 | 12 | 12 | 0.363752 | 0.064 |
| batch | iqp / quimb_cached / 8 | 12 | 12 | 12 | 0.156527 | 0.063 |
| batch | iqp / shared_graph / 1 | 12 | 12 | 12 | 0.0020034 | 0.020 |
| batch | iqp / shared_graph / 128 | 12 | 12 | 12 | 0.0331596 | 0.020 |
| batch | iqp / shared_graph / 32 | 12 | 12 | 12 | 0.00952913 | 0.020 |
| batch | iqp / shared_graph / 8 | 12 | 12 | 12 | 0.00374417 | 0.020 |

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

- Preprocessing `fixed`: median native/strategy time = 2.753× over 45 tasks; candidate faster on 39 and slower on 6.
- Preprocessing `adaptive`: median native/strategy time = 0.794× over 45 tasks; candidate faster on 0 and slower on 45.
- Selector `raw`: median fixed/raw end-to-end time = 26.060× over 42 tasks.
- Selector `effective`: median fixed/effective end-to-end time = 29.424× over 33 tasks.
- Selector `effective_vs_raw`: median raw/effective end-to-end time = 0.905× over 36 tasks.

## Structure and end-to-end cost

Among 48 tasks with rank-step metrics, rank decreased on 0; L increased on 0 of those. Estimated combo increased on 0 of 0 rank-decrease tasks with comparable candidate estimates. Across all comparable tasks, estimated combo increased on 0 of 48; this includes tasks where rank did not decrease.
The examples below compare paired medians for the complete adaptive and native workflows. Adaptive cost includes candidate generation, planning, evaluation, and task overhead; the smaller rank/L is a structural observation, not evidence that the rank change caused the runtime difference.

No rank-decrease task had fully verified matched native and adaptive repeats with adaptive slower in this run set.

## Offline backend-cost oracle regret

These are out-of-fold model predictions compared with the fastest measured backend per task. They are separate from measured online selector replays above. OOF prediction completion is not an independent amplitude verification.

| Policy | OOF tasks | Predictions completed | Median estimated total (s) | Median oracle regret (×) |
|---|---:|---:|---:|---:|
| effective | 48 | 39 | 0.00271115 | 1.000 |
| fixed | 48 | 45 | 0.0926296 | 30.727 |
| raw | 48 | 48 | 0.00216354 | 1.000 |

Selector routes include feature extraction and inference. The PAR2 view assigns each failed, timed-out, unsupported, or unverified replay twice the configured timeout; it complements the verified-pair comparison.

| Selector | Recorded replays | Correctly completed | Completion rate | Total PAR2 (s) |
|---|---:|---:|---:|---:|
| effective | 144 | 108 | 75% | 4322.48 |
| fixed | 144 | 126 | 88% | 2188 |
| raw | 144 | 135 | 94% | 1084.01 |

## Batch reuse by circuit

Each break-even entry is the first tested batch size for which the shared graph run is faster. It is reported only when every repeat for both compared methods passed verification.

| Task | Physical qubits | Unique queries by batch size | First shared win vs independent | First shared win vs Quimb cache |
|---|---:|---|---:|---:|
| graphstate-n10-s0 | 28 | 1: 1, 8: 7, 32: 12, 128: 16 | 1 | 1 |
| graphstate-n4-s0 | 4 | 1: 1, 8: 7, 32: 12, 128: 16 | 1 | 1 |
| graphstate-n6-s0 | 12 | 1: 1, 8: 7, 32: 12, 128: 16 | 1 | 1 |
| graphstate-n8-s0 | 20 | 1: 1, 8: 7, 32: 12, 128: 16 | 1 | 1 |
| iqp-n10-s0 | 10 | 1: 1, 8: 8, 32: 32, 128: 128 | 8 | 1 |
| iqp-n4-s0 | 4 | 1: 1, 8: 8, 32: 31, 128: 121 | 1 | 1 |
| iqp-n6-s0 | 6 | 1: 1, 8: 8, 32: 32, 128: 128 | 1 | 1 |
| iqp-n8-s0 | 8 | 1: 1, 8: 8, 32: 32, 128: 128 | 1 | 1 |

## Reproduction files

The JSONL files preserve individual instances and failures. `summary.json` and the per-experiment CSV files are derived summaries. Re-run the report command after adding run records to rebuild this page and all charts.

This is a small, machine-specific CPU benchmark. A benefit is claimed only for the tested workload, where the measured end-to-end results and correctness checks support it.
