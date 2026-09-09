#!/usr/bin/env bash
#SBATCH -J perturb_F3_whittaker_restart_z0w20
#SBATCH -p mit_normal
#SBATCH -o logs/new_perturbation_F3_whittakerBG12976567_FIXED_TPK_restart_from15517278_rel10_z0wide20.%j.out
#SBATCH -e logs/new_perturbation_F3_whittakerBG12976567_FIXED_TPK_restart_from15517278_rel10_z0wide20.%j.err
#SBATCH --mem=160G
#SBATCH -c 64
#SBATCH --mail-user=jpvelasquez@pucp.edu.pe
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=11:50:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Separation HDF5 selected from the Whittaker/arPLS 16--22 UTC separation.
# Override these environment variables if the HDF5 files are elsewhere.
export SEPARATION_H5="${SEPARATION_H5:-real_whittaker_selected_background_16_22_for_inversion.h5}"

# Background HDF5 selected from the second/restart15 reconstruction using the Whittaker/arPLS separation target.
export BACKGROUND_H5="${BACKGROUND_H5:-new-real-background-inversion_from_sep_16_22_restart15_results_12976567.h5}"

# First perturbation inversion. This restart uses reconstruction/free_params as the new seed
# and fixed/t_pk_s as the inherited peak time.
export PREVIOUS_RESULTS_H5="${PREVIOUS_RESULTS_H5:-new-perturbation-inversion_F3_nexp_whittakerBG12976567_FIXED_TPK_rel10_results_15517278.h5}"

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
    println("[OK] SEPARATION_H5=", get(ENV, "SEPARATION_H5", ""))
    println("[OK] BACKGROUND_H5=", get(ENV, "BACKGROUND_H5", ""))
    println("[OK] PREVIOUS_RESULTS_H5=", get(ENV, "PREVIOUS_RESULTS_H5", ""))
'

echo "[RUN] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
echo "[RUN] Whittaker/arPLS separation target + Whittaker restart15 background 12976567."
echo "[RUN] Restart seed = reconstruction/free_params from first perturbation inversion 15517278."
echo "[RUN] Fixed t_pk inherited from first run; BBO optimizes 6 parameters."
echo "[RUN] Directed bounds: amp,tau_u,tau_d,eps,hz ±10%; z0_km ±20%."

julia --startup-file=no --project=. -t "$JULIA_NUM_THREADS" \
  new-perturbation-data-inversion-bbo_F3_nexp_fixed_TPK_restart_from15517278_whittakerBG12976567_rel10_z0wide20.jl
