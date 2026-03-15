---
name: Data quality before training
description: Always validate data quality with probes before training any model — most failures are data issues, not approach issues
type: feedback
---

Always probe data quality before training. Most past failures were data issues we were unaware of, not bad approaches.

**Why:** Repeated experience of training runs failing or producing misleading results due to upstream data problems (noisy labels, feature disconnects, train/val leaks, format mismatches).

**How to apply:** Before any new training stage, run quick validation probes (correlation checks, memorization baselines, distribution histograms, duplicate checks). See `docs/plans/v6_migration/data_validation.md` for the specific probe checklist. Don't skip this even if the approach seems straightforward.
