#!/usr/bin/env bash
#SBATCH -J bg_polyF3_16_22_degA7Phi4
#SBATCH -p mit_normal
#SBATCH -o logs/new_real_background_inversion_polyF3_simultaneous_16_22_degA7Phi4.%j.out
#SBATCH -e logs/new_real_background_inversion_polyF3_simultaneous_16_22_degA7Phi4.%j.err
#SBATCH --mem=160G
#SBATCH -c 64
#SBATCH --mail-user=jpvelasquez@pucp.edu.pe
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=11:50:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export SEPARATION_H5="${SEPARATION_H5:-simultaneous_polyF3_NAA_V_20080325_window16_22_degA7_degPhi4.h5}"
export INVERSION_SCRIPT="${INVERSION_SCRIPT:-new-real-background-inversion-bbo_from_polyF3_simultaneous_16_22_degA7_degPhi4.jl}"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1

julia --startup-file=no --project=. -t "${JULIA_NUM_THREADS}" "${INVERSION_SCRIPT}"
