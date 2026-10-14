---
name: Bug report
about: Something is broken or produces wrong results
title: "[bug] "
labels: bug, triage
---

## Summary
<!-- One-sentence description of the problem. -->

## Steps to reproduce
```bash
# exact commands you ran, e.g.:
python scripts/phase2_benchmark.py
```

## Expected behaviour
<!-- What you expected to happen. -->

## Actual behaviour
<!-- What actually happened, including error output / stack trace. -->
```
<paste output>
```

## Environment
- OS: <!-- e.g. macOS 14 / Ubuntu 22.04 -->
- Python: <!-- `python -V` -->
- Package version: <!-- `python -c "import nepse_nlp; print(nepse_nlp.__version__)"` -->
- Commit SHA: <!-- `git rev-parse --short HEAD` -->
- Installed extras: <!-- core / deep / api / scrape / track -->

## Reproducibility
- [ ] I ran `make test` and it passes / fails (describe).
- [ ] I can reproduce with `seed=42`.
- [ ] The issue is NOT about the simulated Jev backend (which is a documented stand-in).

## Output of `make pipeline` (if relevant)
<details><summary>phase1->5 output</summary>

```
<paste>
```
</details>
