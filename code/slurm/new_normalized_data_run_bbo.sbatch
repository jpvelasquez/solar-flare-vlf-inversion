#!/usr/bin/env bash
#SBATCH -J new_normalized_inv_data_bbo
#SBATCH -p mit_normal
#SBATCH -o logs/new_normalized_data.%j.out
#SBATCH -e logs/new_normalized_data.%j.err
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

# === Ejecuta tu simulación ===
echo "[RUN] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
julia --startup-file=no --project=. -t "$JULIA_NUM_THREADS" new-real-background-inversion-bbo_from_whittaker_selected_restart15.jl 
#new-real-background-inversion-bbo_from_separation_h5.jl
#new-normalized-data-inversion-bbo.jl

