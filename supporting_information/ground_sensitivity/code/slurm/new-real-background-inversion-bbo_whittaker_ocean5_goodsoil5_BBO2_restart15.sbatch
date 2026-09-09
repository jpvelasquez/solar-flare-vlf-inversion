#!/usr/bin/env bash
#SBATCH -J whit_o5g5_bbo2r15
#SBATCH -p mit_normal
#SBATCH -o logs/whit_o5g5_bbo2r15.%j.out
#SBATCH -e logs/whit_o5g5_bbo2r15.%j.err
#SBATCH --mem=160G
#SBATCH -c 64
#SBATCH --mail-user=jpvelasquez@pucp.edu.pe
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=11:50:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

# Evitar sobresuscripción interna de BLAS/OpenMP.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1
export JULIA_HDF5_USE_SYSTEM_LIBHDF5=0

SCRIPT="new-real-background-inversion-bbo_whittaker_ocean5_goodsoil5_BBO2_restart15.jl"

TARGET_FILE="real_whittaker_selected_background_16_22_for_inversion.h5"

REFERENCE_FILE="outputs_reference_efield_firi_10waveguides_13ground_cases_20080325/reference-efield-firi_ocean5_goodsoil5_10wg_20080325.jld2"

[[ -f "${SCRIPT}" ]] || {
    echo "[ERROR] No se encontró ${SCRIPT} en ${SLURM_SUBMIT_DIR}" >&2
    exit 1
}

[[ -f "${TARGET_FILE}" ]] || {
    echo "[ERROR] No se encontró el target Whittaker: ${TARGET_FILE}" >&2
    exit 1
}

[[ -f "${REFERENCE_FILE}" ]] || {
    echo "[ERROR] No se encontró la referencia FIRI heterogénea: ${REFERENCE_FILE}" >&2
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

echo "======================================================================"
echo "[RUN] Whittaker background inversion — ocean5_goodsoil5 — BBO-2 restart15"
echo "[RUN] SLURM_JOB_ID=${SLURM_JOB_ID:-?}"
echo "[RUN] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
echo "[RUN] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
echo "[RUN] JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
echo "[RUN] OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "[RUN] Script=${SCRIPT}"
echo "[RUN] Target=${TARGET_FILE}"
echo "[RUN] Reference=${REFERENCE_FILE}"
echo "======================================================================"

julia     --startup-file=no     --project=.     --threads="${JULIA_NUM_THREADS}"     "${SCRIPT}"
