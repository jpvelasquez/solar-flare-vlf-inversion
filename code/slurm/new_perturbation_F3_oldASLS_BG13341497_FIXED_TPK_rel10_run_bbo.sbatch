#!/usr/bin/env bash
#SBATCH -J perturb_F3_oldASLS_FIXED_TPK_rel10
#SBATCH -p mit_normal
#SBATCH -o logs/new_perturbation_F3_oldASLS_FIXED_TPK_rel10.%j.out
#SBATCH -e logs/new_perturbation_F3_oldASLS_FIXED_TPK_rel10.%j.err
#SBATCH --mem=160G
#SBATCH -c 64
#SBATCH --mail-user=jpvelasquez@pucp.edu.pe
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=11:50:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Background HDF5 selected from old ASLS restart / second 10 h batch.
# Override this environment variable if the HDF5 is elsewhere.
export BACKGROUND_H5="${BACKGROUND_H5:-new-real-background-inversion-bbo_from_old_asls_restart_results_13341497.h5}"

export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1
export JULIA_HDF5_USE_SYSTEM_LIBHDF5=0

julia --startup-file=no --project=. -e '
    using LinearAlgebra, PyCall, HDF5
    pyimport("numpy"); pyimport("scipy"); pyimport("chaosmagpy")
    println("[OK] Preflight: Threads=", Threads.nthreads(), "  BLAS=", BLAS.get_num_threads())
    println("[OK] BACKGROUND_H5=", get(ENV, "BACKGROUND_H5", ""))
'

echo "[RUN] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
echo "[RUN] Fixed t_pk from data; BBO optimizes 6 parameters only; relative bounds ±10%."

julia --startup-file=no --project=. -t "$JULIA_NUM_THREADS" \
  new-perturbation-data-inversion-bbo_F3_nexp_fixed_TPK_from_data_oldASLS_BG_rel10.jl
