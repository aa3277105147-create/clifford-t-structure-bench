# Examples

## Imported QSAS circuits

These files are copied unchanged from the companion QSAS workbench in this workspace, from its `examples/` directory. They check the QASM import path and give a small GHZ example plus a wide Clifford-only example; neither is part of the generated 48/240 task dataset.

| File | Input characterization | SHA-256 |
|---|---|---|
| `ghz8.qasm` | 8-qubit GHZ preparation with terminal measurements | `afaedc5a7554c06e604308dae7d0c815f3e8e419cc4dae22334f00b968a0a6f6` |
| `clifford512.qasm` | 512-qubit Clifford example with terminal measurements | `6036e172fc70718c9b8f2902b0657ea5f690ace5732b25548161f9097ababa6a` |

The importer removes terminal measurements. The 512-qubit file is a parser and structure example, not a promise that each exact simulator can evaluate it within the local resource budget.

## Structural explanations

Run `python examples/structure_examples.py` to inspect cost transfer under a change of coordinates, a dense effective core behind sparse physical interactions, and bounded-boundary path contraction. These examples check exact equality and explain structure; they do not establish performance gains.

## Custom T/T† batch

Run a user-specified batch of amplitude queries from the repository root:

```bash
python examples/run_batch.py examples/custom_batch.json
```

The JSON file fixes a two-qubit circuit, input and output basis states, global phase, memory and time budgets, and four fillings for its two T/T† slots. Edit `sign_vectors` to supply other `+1` (T) and `-1` (T†) rows. The command calls the same resource-limited `run_job` API as the benchmark and prints the amplitudes, task hash, status, timings, and peak RSS. It sets `verification` to `not_checked`; compare the results with a reference before treating them as verified.
