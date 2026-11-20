# Explicit revalidation of an existing checkpoint

Use a new candidate record when an existing checkpoint is evaluated again.
Preserve the previous model version and its recorded metrics. A changed
candidate must never silently replace the earlier evidence.

The `register` command accepts two additional options:

```text
--revalidate-version PREVIOUS_VERSION
--revalidation-reason "Reason for the new evaluation"
```

All ordinary registration inputs are still required. The CLI recomputes the
candidate from the archive, checkpoint, manifest, counting files and evaluator
environment before any registration. No training is performed.

The explicit predecessor must have passed inference verification. Its original
and imported run IDs, checkpoint, manifest, export archive, architecture and
training provenance must match the new candidate. The new evidence must pass
the full validation contract and include verified comparison evidence.

The new model version records `parkvision.revalidation_of` and
`parkvision.revalidation_reason`. Older versions, imported training runs and
aliases are left unchanged. An exact-version load and prediction must succeed
before `parkvision.inference_verified` becomes true.

Run the same explicit command again after an interrupted inference check to
verify and reuse the matching version. Conflicting sibling evidence or multiple
matching versions cause an error requiring inspection. If the first attempt
failed before version creation, a separate packaging run can remain; inspect
remote state after network failures. This is not a distributed transaction.

Coordinate one registry writer across the team. The existing file lock only
coordinates processes that actually share the same lock file.

For ParkVision C, version 1 retains the historical MAE occupied of 0.1496 and
exact-count accuracy of 0.7612. The fresh validation reports 0.1500 and 0.7608.
These are different evaluation records for the same checkpoint. The new
version number must be obtained from the server, never assumed to be 3.

After registration, archive the candidate, counting CSV, counting summary,
evaluation environment, quality configuration and evaluation protocol under
`validation_evidence/` in the new packaging run. Download them and check their
hashes. Do not attach these new files as if they were the original version's
evaluation evidence.

Revalidation does not assign `champion`, enable automatic promotion or deploy
an API. Model A remains an MLflow baseline if it fails the quality thresholds.

Validation of this change: 96 tests and 48 unittest subtests passed locally
under Python 3.12 with MLflow Skinny 3.17.2; Ruff and Python compilation checked.
This local result does not replace the project's full MLflow CI matrix or a
real DagsHub inference check.
