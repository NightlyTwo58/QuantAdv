# Active source layout

- `config.py` defines dataset, model, attack, quantization, and output settings.
- `QuantAdv.py` builds model variants, runs enabled experiments, and saves
  resumable results.
- `quantization.py` implements fake-quantized layers, PTQ/QAT helpers, and
  straight-through gradient controls.
- `attack.py` contains attacks, evaluation helpers, and attack diagnostics;
  `attack_cache.py` caches reusable attack results.
- `defense.py` contains optional defenses and their evaluation helpers.
- `stats.py` computes confidence intervals and paired correctness tests;
  `ResourceMonitor.py` records run resource usage.
- `graphs/` merges partial outputs and creates summary or formal figures.

Run from the repository root:

```powershell
python -m src.QuantAdv
python -m src.graphs.combine
python -m src.graphs.formaldata
```

Experiment behavior is controlled in `config.py`. Outputs and datasets are
stored under the project-level `data/` and dataset directories.
