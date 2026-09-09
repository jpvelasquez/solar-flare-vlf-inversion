# Provenance caveats

## ASLS preprocessing (`datos_asls_full-v3.h5`)

The canonical ASLS file in this repository is `data/background_separation/datos_asls_full-v3.h5`. It is the version that matches the targets stored in the final ASLS inversion chain.

The available local notebook `notebooks/preprocessing/PyBaselines-data-Copy1.ipynb` is a later descendant of the notebook used to create that file. The user reports that the output name was changed from `v3` to `v4` to avoid overwriting the previous HDF5. The notebook was also subsequently edited, so its present saved state should be treated as method lineage rather than an exact historical execution snapshot. The canonical HDF5 itself, the downstream inversion outputs, and the Engaging logs are preserved.

## ASLS background restart job 13341497

The exact pre-run `.jl` snapshot for job 13341497 was not preserved. The later supplied `new-real-background-inversion-bbo_from_old_asls_restart.jl` had been edited to restart from job 13341497 for a subsequent run. The job 13341497 stdout explicitly reports:

`Restart seed source: new-real-background-inversion-bbo_from_old_asls_results_13256787.h5`

The repository therefore includes:

- `code/julia/canonical_main/new-real-background-inversion-bbo_from_old_asls_restart_RECONSTRUCTED_job13341497.jl`: a transparent reconstruction in which only the previous-results source and matching HDF5 provenance attribute were restored to job 13256787; and
- `provenance/post_run_snapshots/new-real-background-inversion-bbo_from_old_asls_restart_POSTRUN_MODIFIED.jl`: the later source exactly as supplied.

The reconstruction is explicitly labeled and is not claimed to be a byte-identical historical snapshot.

## Slurm launchers

Several `.sbatch` files were reused and edited between runs. The authoritative mapping between jobs, scripts, and output HDF5 files is `metadata/CANONICAL_RUNS.tsv`, supported by `provenance/job_logs/`. Launcher snapshots are retained for cluster configuration and execution context but should not be used alone to infer which code revision produced a specific job.

## Python environment

Exact historical Python package versions were not captured. The provided minimal requirements file lists the imported packages only. The Julia environment is pinned by `Project.toml` and `Manifest.toml`.

## Generated figures

Only a subset of already-rendered figures is bundled. The numerical data and figure-generation notebooks are the authoritative reproducibility products; rendered figures can be regenerated from those sources.
