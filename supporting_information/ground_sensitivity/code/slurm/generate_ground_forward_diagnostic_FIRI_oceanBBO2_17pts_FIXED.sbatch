#!/usr/bin/env bash
#SBATCH -J ground_fwd_diag17_fix
#SBATCH -p mit_normal
#SBATCH -o logs/ground_fwd_diag17_fix.%j.out
#SBATCH -e logs/ground_fwd_diag17_fix.%j.err
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

SCRIPT="generate_ground_forward_diagnostic_FIRI_oceanBBO2_17pts_FIXED.jl"

export WHITTAKER_TARGET_H5="${WHITTAKER_TARGET_H5:-${SLURM_SUBMIT_DIR}/real_whittaker_selected_background_16_22_for_inversion.h5}"

export OCEAN_BBO2_H5="${OCEAN_BBO2_H5:-${SLURM_SUBMIT_DIR}/new-real-background-inversion_from_sep_16_22_restart15_results_12976567.h5}"

export REFERENCE_DIR="${REFERENCE_DIR:-${SLURM_SUBMIT_DIR}/outputs_reference_efield_firi_10waveguides_13ground_cases_20080325}"

REFERENCE_OCEAN="${REFERENCE_DIR}/reference-efield-firi_ground_10_10wg_20080325.jld2"
REFERENCE_7_3="${REFERENCE_DIR}/reference-efield-firi_ocean7_goodsoil3_10wg_20080325.jld2"
REFERENCE_6_4="${REFERENCE_DIR}/reference-efield-firi_ocean6_goodsoil4_10wg_20080325.jld2"
REFERENCE_5_5="${REFERENCE_DIR}/reference-efield-firi_ocean5_goodsoil5_10wg_20080325.jld2"

[[ -f "${SCRIPT}" ]] || {
    echo "[ERROR] No se encontró ${SCRIPT} en ${SLURM_SUBMIT_DIR}" >&2
    exit 1
}

[[ -f "${WHITTAKER_TARGET_H5}" ]] || {
    echo "[ERROR] No se encontró el target Whittaker: ${WHITTAKER_TARGET_H5}" >&2
    exit 1
}

[[ -f "${OCEAN_BBO2_H5}" ]] || {
    echo "[ERROR] No se encontró la inversión oceánica BBO-2: ${OCEAN_BBO2_H5}" >&2
    exit 1
}

[[ -d "${REFERENCE_DIR}" ]] || {
    echo "[ERROR] No se encontró el directorio de referencias: ${REFERENCE_DIR}" >&2
    exit 1
}

for reference_file in \
    "${REFERENCE_OCEAN}" \
    "${REFERENCE_7_3}" \
    "${REFERENCE_6_4}" \
    "${REFERENCE_5_5}"
do
    [[ -f "${reference_file}" ]] || {
        echo "[ERROR] No se encontró la referencia FIRI: ${reference_file}" >&2
        exit 1
    }
done

[[ -f "mytools/preamble.jl" ]] || {
    echo "[ERROR] No se encontró mytools/preamble.jl" >&2
    exit 1
}

[[ -f "mytools/scenario.jl" ]] || {
    echo "[ERROR] No se encontró mytools/scenario.jl" >&2
    exit 1
}

echo "================================================================================"
echo "[RUN] Forward-only ground diagnostic — FIRI hardcoded + ocean BBO-2"
echo "[RUN] Fixed target offsets: +19 dB, -210 deg for every case"
echo "[RUN] Ground cases: ocean10, ocean7_goodsoil3, ocean6_goodsoil4, ocean5_goodsoil5"
echo "[RUN] SLURM_JOB_ID=${SLURM_JOB_ID:-?}"
echo "[RUN] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
echo "[RUN] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
echo "[RUN] JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
echo "[RUN] OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "[RUN] Script=${SCRIPT}"
echo "[RUN] Target=${WHITTAKER_TARGET_H5}"
echo "[RUN] Ocean BBO-2=${OCEAN_BBO2_H5}"
echo "[RUN] References=${REFERENCE_DIR}"
echo "================================================================================"

julia \
    --startup-file=no \
    --project=. \
    --threads="${JULIA_NUM_THREADS}" \
    "${SCRIPT}"
