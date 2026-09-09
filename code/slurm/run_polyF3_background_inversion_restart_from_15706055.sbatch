#!/usr/bin/env bash
#SBATCH -J polyF3_bg_bbo2
#SBATCH -p mit_normal
#SBATCH -o logs/new_polyF3_background_inversion_restart15706055.%j.out
#SBATCH -e logs/new_polyF3_background_inversion_restart15706055.%j.err
#SBATCH --mem=160G
#SBATCH -c 64
#SBATCH --mail-user=jpvelasquez@pucp.edu.pe
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=11:50:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# === Depot local del proyecto (persistente y aislado) ===
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot"

# === Hilos coherentes: Julia paraleliza; BLAS NO ===
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1

# === Usar JLL para HDF5 (evita choques con libs del sistema) ===
export JULIA_HDF5_USE_SYSTEM_LIBHDF5=0

# === Sanity check mínimo (no instala nada; falla temprano si falta algo) ===
julia --startup-file=no --project=. -e '
    using LinearAlgebra, PyCall, HDF5
    pyimport("numpy"); pyimport("scipy"); pyimport("chaosmagpy");
    println("[OK] Preflight: Threads=", Threads.nthreads(), "  BLAS=", BLAS.get_num_threads())
'

# === Ejecuta la inversión PolyF3 background BBO-2 restart desde job 15706055 ===
echo "[RUN] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
julia --startup-file=no --project=. -t "$JULIA_NUM_THREADS" new-real-background-inversion-bbo_from_polyF3_simultaneous_restart_from_15706055.jl
