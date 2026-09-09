#!/usr/bin/env bash
#SBATCH -J firi_ref10_13ground
#SBATCH -p mit_normal
#SBATCH -o logs/firi_ref10_13ground.%j.out
#SBATCH -e logs/firi_ref10_13ground.%j.err
#SBATCH --mem=160G
#SBATCH -c 64
#SBATCH --mail-user=jpvelasquez@pucp.edu.pe
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=11:50:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot"

# Distributed parallelism:
# 1 coordinating process + 63 one-thread local workers.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1
export JULIA_HDF5_USE_SYSTEM_LIBHDF5=0

SCRIPT="run_reference_efield_firi_10wg_13ground_cases.jl"

[[ -f "${SCRIPT}" ]] || {
    echo "[ERROR] No se encontró ${SCRIPT} en ${SLURM_SUBMIT_DIR}" >&2
    exit 1
}

[[ -f "mytools/preamble.jl" ]] || {
    echo "[ERROR] No se encontró mytools/preamble.jl" >&2
    exit 1
}

[[ -f "mytools/scenario.jl" ]] || {
    echo "[ERROR] No se encontró mytools/scenario.jl" >&2
    exit 1
}

julia --startup-file=no --project=. -e '
    using LinearAlgebra, JLD2, DSP, Interpolations, LongwaveModePropagator
    BLAS.set_num_threads(1)
    println("[OK] Preflight: Threads=", Threads.nthreads(),
            "  BLAS=", BLAS.get_num_threads(),
            "  GROUND=", length(LongwaveModePropagator.GROUND))
'

echo "[RUN] SLURM_JOB_ID=${SLURM_JOB_ID:-?}"
echo "[RUN] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
echo "[RUN] JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
echo "[RUN] Script=${SCRIPT}"
echo "[RUN] Existing compatible checkpoint will be resumed automatically."

julia --startup-file=no --project=. -t 1 "${SCRIPT}"
