#!/usr/bin/env bash
set -euo pipefail

# Recopila los archivos HDF5 necesarios para auditar y comparar
# las inversiones concatenadas de perturbación:
# Whittaker, ASLS con t_pk fijo y background polinómico.
#
# Uso:
#   ./recopilar_h5_inversiones_perturbacion.sh
#   ./recopilar_h5_inversiones_perturbacion.sh /ruta/al/proyecto
#   ./recopilar_h5_inversiones_perturbacion.sh /ruta/al/proyecto /ruta/de/salida
#
# Por defecto:
#   - busca desde el directorio actual;
#   - crea una carpeta fechada dentro del directorio del proyecto;
#   - conserva la ruta relativa de cada archivo;
#   - no mueve ni modifica los originales.

ROOT_INPUT="${1:-.}"
ROOT="$(realpath "$ROOT_INPUT")"

if [[ ! -d "$ROOT" ]]; then
    printf 'ERROR: el directorio del proyecto no existe: %s\n' "$ROOT_INPUT" >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_INPUT="${2:-$ROOT/h5_auditoria_inversiones_perturbacion_${timestamp}}"
mkdir -p "$OUT_INPUT"
OUTDIR="$(realpath "$OUT_INPUT")"

if [[ "$OUTDIR" == "$ROOT" ]]; then
    printf 'ERROR: la carpeta de salida no puede ser el directorio raíz del proyecto.\n' >&2
    exit 1
fi

MANIFEST="$OUTDIR/manifest_h5.tsv"
MISSING="$OUTDIR/archivos_faltantes.txt"
MULTIPLE="$OUTDIR/coincidencias_multiples.txt"
SUMMARY="$OUTDIR/resumen.txt"

printf 'familia\titem\tbusqueda_usada\tarchivo_origen\tcopia_destino\tsha256\n' > "$MANIFEST"
: > "$MISSING"
: > "$MULTIPLE"

# Formato:
# familia|identificador|nombre exacto esperado|patrón alternativo
#
# El patrón alternativo solo se usa cuando no aparece el nombre exacto.
# Es especialmente importante para el primer resultado ASLS FIXED_TPK,
# cuyo nombre fue inferido a partir del restart 15373069.
RECORDS=(
'Whittaker|perturbacion_BBO1_15517278|new-perturbation-inversion_F3_nexp_whittakerBG12976567_FIXED_TPK_rel10_results_15517278.h5|*whittakerBG12976567*FIXED_TPK*results_15517278.h5'
'Whittaker|perturbacion_BBO2_15572093|new-perturbation-inversion_F3_nexp_whittakerBG12976567_FIXED_TPK_restart_from15517278_rel10_z0wide20_results_15572093.h5|*whittakerBG12976567*FIXED_TPK*results_15572093.h5'
'Whittaker|background_BBO2_12976567|new-real-background-inversion_from_sep_16_22_restart15_results_12976567.h5|*background*sep_16_22*results_12976567.h5'
'Whittaker|datos_background_seleccionado|real_whittaker_selected_background_16_22_for_inversion.h5|*whittaker*selected*background*16_22*.h5'

'ASLS_FIXED_TPK|perturbacion_BBO1_15341092|new-perturbation-inversion_F3_nexp_oldASLS_BG13341497_FIXED_TPK_rel10_results_15341092.h5|*oldASLS*BG13341497*FIXED_TPK*results_15341092.h5'
'ASLS_FIXED_TPK|perturbacion_BBO2_15373069|new-perturbation-inversion_F3_nexp_oldASLS_BG13341497_FIXED_TPK_rel10_seed_from_15341092_results_15373069.h5|*oldASLS*BG13341497*FIXED_TPK*results_15373069.h5'
'ASLS_FIXED_TPK|background_13341497|new-real-background-inversion-bbo_from_old_asls_restart_results_13341497.h5|*old_asls*restart*results_13341497.h5'
'ASLS_FIXED_TPK|datos_ASLS|datos_asls_full-v3.h5|datos_asls_full-v3.h5'

'PolyF3|perturbacion_BBO1_15824988|new-perturbation-inversion_F3_nexp_polyF3BG15742708_FIXED_TPK_rel10_results_15824988.h5|*polyF3BG15742708*FIXED_TPK*results_15824988.h5'
'PolyF3|perturbacion_BBO2_15868514|new-perturbation-inversion_F3_nexp_polyF3BG15742708_FIXED_TPK_restart_from_15824988_rel10_results_15868514.h5|*polyF3BG15742708*FIXED_TPK*results_15868514.h5'
'PolyF3|background_15742708|new-real-background-inversion_from_polyF3_simultaneous_restart_from_15706055_results_15742708.h5|*polyF3*background*results_15742708.h5'
'PolyF3|datos_separacion_polinomica|simultaneous_polyF3_NAA_V_20080325_window16_22_degA7_degPhi4.h5|*simultaneous_polyF3*NAA_V*20080325*window16_22*.h5'

'Series_5min|ASLS_Whittaker|series_5min_seed_oldASLSbest_whittakerbest_FIXED_TPK.h5|*series_5min*oldASLSbest*whittakerbest*FIXED_TPK.h5'
'Series_5min|PolyF3|perturbation_polyF3BG15742708_seed_BBO1_15824988_BBO2_15868514_timeseries_5min.h5|*polyF3BG15742708*15824988*15868514*5min.h5'
)

found_items=0
missing_items=0
ambiguous_items=0
copied_files=0

printf 'Directorio del proyecto: %s\n' "$ROOT"
printf 'Carpeta de salida:       %s\n\n' "$OUTDIR"

for record in "${RECORDS[@]}"; do
    IFS='|' read -r family item exact_name fallback_pattern <<< "$record"

    matches=()
    search_used="$exact_name"

    # Primero se exige el nombre exacto.
    mapfile -d '' -t matches < <(
        find "$ROOT" \
            \( -path "$OUTDIR" -o -path "$OUTDIR/*" \) -prune -o \
            -type f -name "$exact_name" -print0
    )

    # Si no aparece, se intenta un patrón alternativo controlado.
    if (( ${#matches[@]} == 0 )) && [[ -n "$fallback_pattern" ]]; then
        search_used="$fallback_pattern"
        mapfile -d '' -t matches < <(
            find "$ROOT" \
                \( -path "$OUTDIR" -o -path "$OUTDIR/*" \) -prune -o \
                -type f -iname "$fallback_pattern" -print0
        )
    fi

    if (( ${#matches[@]} == 0 )); then
        printf '[FALTA] %-16s %s\n' "$family" "$item"
        printf '%s\t%s\texacto=%s\talternativo=%s\n' \
            "$family" "$item" "$exact_name" "$fallback_pattern" >> "$MISSING"
        missing_items=$((missing_items + 1))
        continue
    fi

    found_items=$((found_items + 1))

    if (( ${#matches[@]} > 1 )); then
        ambiguous_items=$((ambiguous_items + 1))
        {
            printf '%s | %s | %d coincidencias | búsqueda: %s\n' \
                "$family" "$item" "${#matches[@]}" "$search_used"
            printf '  %s\n' "${matches[@]}"
            printf '\n'
        } >> "$MULTIPLE"
        printf '[MÚLTIPLE: %d] %-10s %s\n' "${#matches[@]}" "$family" "$item"
    else
        printf '[OK] %-19s %s\n' "$family" "$item"
    fi

    # Se copian todas las coincidencias y se conserva la ruta relativa
    # para impedir colisiones entre archivos homónimos.
    for src in "${matches[@]}"; do
        rel="${src#"$ROOT"/}"
        dest="$OUTDIR/$family/$rel"

        mkdir -p "$(dirname "$dest")"
        cp -a -- "$src" "$dest"

        checksum="$(sha256sum "$dest" | awk '{print $1}')"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$family" "$item" "$search_used" "$src" "$dest" "$checksum" \
            >> "$MANIFEST"

        copied_files=$((copied_files + 1))
    done
done

{
    printf 'Auditoría de archivos HDF5 para las inversiones de perturbación\n'
    printf '===============================================================\n'
    printf 'Proyecto:                 %s\n' "$ROOT"
    printf 'Salida:                   %s\n' "$OUTDIR"
    printf 'Ítems solicitados:        %d\n' "${#RECORDS[@]}"
    printf 'Ítems encontrados:        %d\n' "$found_items"
    printf 'Ítems faltantes:          %d\n' "$missing_items"
    printf 'Ítems con coincidencias:  %d\n' "$ambiguous_items"
    printf 'Archivos copiados:        %d\n' "$copied_files"
    printf '\n'
    printf 'Inventario y checksums:   %s\n' "$MANIFEST"
    printf 'Faltantes:                %s\n' "$MISSING"
    printf 'Coincidencias múltiples:  %s\n' "$MULTIPLE"
} | tee "$SUMMARY"

printf '\nProceso terminado.\n'

if (( missing_items > 0 )); then
    printf 'ADVERTENCIA: faltan %d ítems. Revisa %s\n' "$missing_items" "$MISSING" >&2
    exit 2
fi
