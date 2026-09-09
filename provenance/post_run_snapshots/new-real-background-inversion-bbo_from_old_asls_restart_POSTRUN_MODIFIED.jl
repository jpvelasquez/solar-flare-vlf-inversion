#####################################################################################
# Inversión del background con datos reales usando el background antiguo ASLS
# Restart: usa como semilla la reconstrucción BBO previa del job 13256787
# Target: datos_asls_full-v3.h5
# Loss: chi2 normalizado A + phi, sin pesos extra.
#####################################################################################

include("./mytools/preamble.jl")
include("./mytools/scenario.jl")

using Dates
using TimeZones
using HDF5
using JLD2
using DSP
using AstroLib
using Polynomials
using LinearAlgebra
using Statistics
using BlackBoxOptim
import Interpolations
using Interpolations: LinearInterpolation

# 0) PREP: cache de constantes
###############################
Base.@kwdef struct SimCache
    tx::Any
    rx::Any
    rx_rec::Any
    ground::Any
    dists12::Vector{Float64}   # dists_new[1:2:end]
    dists22::Vector{Float64}   # dists_new[2:2:end]
    dt::Vector{ZonedDateTime}  # ZonedDateTime.(unix2datetime.(ts), tz"UTC")
    bf_list::Vector{Any}       # campo magnético precomputado por tiempo
end

function build_cache(ts, tx, rx, rx_rec, ground, dists_new)
    dt = ZonedDateTime.(unix2datetime.(ts), tz"UTC")
    d12 = dists_new[1:2:end]
    d22 = dists_new[2:2:end]
    bf_list = [igrf(tx, rx_rec, year(ti), d22; alt=60e3) for ti in dt]  # una vez
    return SimCache(tx=tx, rx=rx, rx_rec=rx_rec, ground=ground,
                    dists12=d12, dists22=d22, dt=dt, bf_list=bf_list)
end

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vlfsignal_simple_perturbation_v2(zdt_test_ref,ϕ_firi_wrapped_ref,mat_sza, pars_list, cache::SimCache)
    p_h_sza = Polynomial(pars_list[1])
    p_β_sza = Polynomial(pars_list[2])

    Nt = length(cache.dt)
    A_fitₜ = Vector{Float64}(undef, Nt)   # sin push!
    ϕ_fitₜ = Vector{Float64}(undef, Nt)

    @inbounds for k in 1:Nt
        sza_tᵢ      = mat_sza[k, :]
        rad_sza_tᵢ  = (π/180.0) .* sza_tᵢ
        hs_aux      = p_h_sza.(rad_sza_tᵢ[2:2:end])
        βs_aux      = p_β_sza.(rad_sza_tᵢ[2:2:end])
        Nwaveguide  = length(sza_tᵢ[2:2:end])

        bf_k        = cache.bf_list[k]            # ya precomputado
        species2    = Vector{Any}(undef, Nwaveguide)
        for i in 1:Nwaveguide
            h_i, β_i = hs_aux[i], βs_aux[i]
            species2[i] = Species(QE, ME, z->waitprofile(z, h_i, β_i), electroncollisionfrequency)
        end

        wg2 = SegmentedWaveguide([HomogeneousWaveguide(bf_k[j], species2[j],
                  cache.ground, cache.dists12[j]) for j in 1:Nwaveguide])

        _, A_fit_seg, ϕ_fit_seg = propagate(wg2, cache.tx, cache.rx)
        A_fitₜ[k] = Float64(A_fit_seg)
        ϕ_fitₜ[k] = Float64(ϕ_fit_seg)
    end
    ϕ_firi_unwrapped = unwrap_resampled(zdt_test_ref,ϕ_firi_wrapped_ref, cache.dt,  ϕ_fitₜ)
    #ϕ_fitₜ = rad2deg.(unwrap(ϕ_fitₜ))
    return A_fitₜ, rad2deg.(ϕ_firi_unwrapped)
end

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


###########################################################################################

# save_results_h5/load_results_h5 del flujo antiguo fueron retirados.
# El guardado activo de este script usa save_real_background_inversion_h5 más abajo.

########################################################################

"""
    chi_signed_astrolib(lat_deg, lon_deg, zdt; height_m=0.0, units=:deg)

Ç con signo estilo LWPC:
- Ç<0 (Este/AM), Ç>0 (Oeste/PM).
- `units`=:deg o :rad.
"""
function chi_signed_astrolib(lat_deg::Real, lon_deg::Real, zdt::ZonedDateTime;
                             height_m::Real=0.0, units::Symbol=:deg)
    zdt_utc = astimezone(zdt, tz"UTC")
    jd  = AstroLib.jdcnv(DateTime(zdt_utc))
    ra, dec = AstroLib.sunpos(jd)                           # grados
    alt_deg, az_deg, _ha_deg = AstroLib.eq2hor(ra, dec, jd, lat_deg, lon_deg, height_m)
    sza = 90.0 - alt_deg                                    # SZA sin signo (grados)
    # Convención eq2hor: az=0 N, 90 E, 180 S, 270 W
    #chi_deg = (0.0 d az_deg < 180.0) ? -sza : +sza          # Estenegativo, Oestepositivo
    chi_deg = (0.0 <= az_deg < 180.0) ? -sza : +sza          # Estenegativo, Oestepositivo

    return units === :rad ? deg2rad(chi_deg) : chi_deg
end

#####################################################################

const LinBC = Interpolations.Linear()

function unwrap_resampled(
    t_ref::Vector{ZonedDateTime}, phi_wr_ref::AbstractVector{<:Real},
    t_new::Vector{ZonedDateTime}, phi_wr_new::AbstractVector{<:Real};
    discont = 0.9*π
)
    @assert length(t_ref) == length(phi_wr_ref)
    @assert length(t_new) == length(phi_wr_new)
    @assert issorted(t_ref)
    @assert issorted(t_new)

    # 1) Unwrap de la serie ORIGINAL (rad)
    phi_ref_unw = DSP.unwrap(collect(float.(phi_wr_ref)); dims=1, discont=discont, period=2π)

    # 2) Numerizar tiempos en UTC (ms desde época)
    to_ms(zdt_vec) = Float64.(Dates.value.(DateTime.(astimezone.(zdt_vec, Ref(tz"UTC")))))
    x_ref = to_ms(t_ref)
    x_new = to_ms(t_new)

    # 3) Interpolar la referencia unwrapped a los tiempos nuevos
    itp_phi = LinearInterpolation(x_ref, phi_ref_unw; extrapolation_bc=LinBC)
    phi_ref_at_new = itp_phi.(x_new)

    # 4) Alinear cada muestra nueva a la rama más cercana de la referencia
    phi_wr_new_f = collect(float.(phi_wr_new))
    phi_unw_new  = similar(phi_wr_new_f)
    @inbounds for k in eachindex(phi_wr_new_f)
        m = round((phi_ref_at_new[k] - phi_wr_new_f[k]) / (2*π))
        phi_unw_new[k] = phi_wr_new_f[k] + m*2*π
    end
    return phi_unw_new
end
#############################################################################################################################
################################   Cargando fase de referencia de FIRI para wl unwrapping  ##################################
##############################################################################################################################
filename_ref = "./reference-efield-firi.jld2"
jldfile_ref = jldopen(filename_ref, "r")
A_firi_ref = jldfile_ref["amp"]
phi_firi_wrapped_ref = jldfile_ref["phi"]
zdt_ref = jldfile_ref["zdt"]
close(jldfile_ref)
##############################################################################################################################
############################################## Cargando los datos   ##########################################################
##############################################################################################################################
##############################################################################################################################
#################### Cargando HDF5 ASLS antiguo para inversión de background ###############################################
##############################################################################################################################

# Archivo antiguo generado con Pybaselines/ASLS.
# Contiene el intervalo 16:00–22:00 UT a resolución de 1 s.
# IMPORTANTE: la inversión sigue usando solamente 18:00–22:00 UT mediante zdt_test e indices.
filename_sep = "datos_asls_full-v3.h5"

# Lectura directa del esquema antiguo. No hay fallbacks para otros nombres.
timestamps_unix_sep = Float64.(vec(h5read(filename_sep, "time_unix")))
time_datetime = unix2datetime.(timestamps_unix_sep)
time_zoned = ZonedDateTime.(time_datetime, tz"UTC")
timestamps_iso = string.(time_zoned)

# Background antiguo ASLS.
A_background_sep = Float64.(vec(h5read(filename_sep, "baseline_asls_amp")))
phi_background_sep = Float64.(vec(h5read(filename_sep, "baseline_asls_phi")))

# Perturbación antigua ASLS = data - background.
A_perturbation_sep = Float64.(vec(h5read(filename_sep, "detrended_asls_amp")))
phi_perturbation_sep = Float64.(vec(h5read(filename_sep, "detrended_asls_phi")))

# El archivo antiguo no guarda explícitamente A_real/phi_real ni total_fit.
# Se reconstruyen solo para diagnóstico como background + perturbation.
# La loss usa únicamente A_background_sep y phi_background_sep.
A_real_sep = A_background_sep .+ A_perturbation_sep
phi_real_sep = phi_background_sep .+ phi_perturbation_sep
A_total_fit_sep = copy(A_real_sep)
phi_total_fit_sep = copy(phi_real_sep)

# Compatibilidad con la lógica original del script:
# esta inversión ajusta SOLO el background.
baseline_amp = A_background_sep
baseline_phi = phi_background_sep
amp_detrended = A_perturbation_sep
phi_detrended = phi_perturbation_sep

# Verificación mínima de consistencia dimensional.
nsep = length(time_zoned)
arrays_to_check = Dict(
    "A_real_reconstructed" => A_real_sep,
    "phi_real_reconstructed" => phi_real_sep,
    "baseline_asls_amp" => A_background_sep,
    "baseline_asls_phi" => phi_background_sep,
    "detrended_asls_amp" => A_perturbation_sep,
    "detrended_asls_phi" => phi_perturbation_sep,
)

for (name, arr) in arrays_to_check
    if length(arr) != nsep
        error("Dataset $(name) tiene longitud $(length(arr)); time_unix tiene longitud $(nsep).")
    end
end

println("Archivo de separación ASLS antiguo cargado: ", filename_sep)
println("Número de muestras cargadas: ", nsep)
println("Rango temporal del archivo:")
println("  ", first(time_zoned))
println("  ", last(time_zoned))
println("Datasets usados:")
println("  time_unix")
println("  baseline_asls_amp")
println("  baseline_asls_phi")
println("  detrended_asls_amp")
println("  detrended_asls_phi")
println("A_real/phi_real fueron reconstruidos solo para diagnóstico como baseline + detrended.")

time_dt = Date.(time_zoned)
#################################################################################################################################
#####################################
######Generando arreglos de timepo###
z = collect(alt)/1e3; # 151 vector
t_step = Minute(5)
#zdt0 = ZonedDateTime(2011, 2, 13, 16, 0, 0, tz"UTC") # Simulation starting time
zdt0 = ZonedDateTime(2008, 3, 25, 18, 0, 0, tz"UTC") # Simulation starting time
zdt = datelinspace(zdt0, zdt0+Hour(4), t_step)
#zdt_new = datelinspace(zdt0, zdt0+Hour(24), t_step)
ztimerange = copy(zdt)
timerange = DateTime.(ztimerange)
Nt = length(timerange)
### Generando trayectorias y objeto "Receiver" para usar IGRF
npoints = 19#5#6#3#12#20
lats_new, lons_new, dists_new = geolinspace("NAA", "PLO", npoints, false)
lats_new2, lons_new2, dists_new2 = geolinspace("NAA", "PIU", npoints, false)
N = npoints
#n = N + 2
lon_rx, lat_rx = rx_params["Longitude"], rx_params["Latitude"]#P2[1], P2[2]
lon_rx2, lat_rx2 = rx_params2["Longitude"], rx_params2["Latitude"]#P3[1], P3[2]
rx_rec = Receiver("PLO",lat_rx, lon_rx, 0.0, VerticalDipole())
println("Number of waveguide segments: ", length(lats_new[2:2:end]))
### Re-sampling de los valores de tiempo:
zdt_test = zdt[1:3:end]#[1:3:end]
timerange_test = DateTime.(zdt_test)
timestamps_test = datetime2unix.(timerange_test)
println(length(zdt_test))

### Buscar qué índices del arreglo de tiempo de los datos corresponden a los valores del arreglo de tiempo re-sampleado.
index_map = Dict(t => i for (i, t) in enumerate(time_zoned))

function nearest_time_index(tvec::Vector{ZonedDateTime}, t::ZonedDateTime)
    Δ = abs.(Dates.value.(DateTime.(tvec) .- DateTime(t)))
    return argmin(Δ)
end

indices = Vector{Int}(undef, length(zdt_test))
for (j, tt) in enumerate(zdt_test)
    idx = get(index_map, tt, nothing)
    if idx === nothing
        idx = nearest_time_index(time_zoned, tt)
        println("$(tt)  →  no se encontró exacto; usando índice más cercano $(idx) ($(time_zoned[idx]))")
    else
        println("$(tt)  →  índice $(idx)")
    end
    indices[j] = idx
end

# Verificación explícita: el archivo nuevo empieza antes (16:00),
# pero la inversión debe seguir usando 18:00–22:00.
println("Primer tiempo de inversión: ", first(zdt_test), " -> índice ", first(indices), " / archivo: ", time_zoned[first(indices)])
println("Último tiempo de inversión:  ", last(zdt_test),  " -> índice ", last(indices),  " / archivo: ", time_zoned[last(indices)])

mat_sza = Matrix{Float64}(undef,size(zdt_test)[1],size(dists_new)[1])

for (k,tt) in enumerate(zdt_test)
    #sza_temp = get_sza.(lats_new, lons_new, tt)
    sza_temp = chi_signed_astrolib.(lats_new, lons_new, tt)#
    mat_sza[k,:] = sza_temp
end
##############################################################
### Semilla FIRI: h' de grado 8 y β de grado 5.
###############################################################
# Coeficientes tomados de la tesis de Alberto
#coeffs_h_alb = #[70.55,0.045,1.27,-0.92,3.715,0.295,-1.491,-0.122,0.268]
#coeffs_β_alb = [0.395,0.005,-0.046,0.0075,-0.0215,-0.0024]
### Los valores de los coeficientes fueron tomados de simular en 
### las mismas condiciones con FIRI
coeffs_β_alb =  [0.383338637221558,-0.0020295433454318493,0.03247156108586324,0.006262402864343245,-0.03719773552870514, -0.0038082733033810846]
coeffs_h_alb = [70.93157100832119,0.02362601104815065,0.6309437477618507,-0.1172361567740213,1.781775057195727,0.1798626527735148,-0.7183163078548129,-0.07238044623458698,0.29364907405058155]
#####################################
# Coefficients from a first 12 hours batch 
#coeffs_h_alb = [71.40417159309351, 0.02357940411008864, 0.618595137817641, -0.11625667818013954, 1.8088199542297674, 0.17626648038888348, -0.7326815338775464, -0.07108069863572547, 0.28780236570576834] 
#coeffs_β_alb = [0.3773484943364485, -0.00205008458608453, 0.032345338916790596, 0.006182760492648512, -0.037559311633157066, -0.0038834251133583945]


#Coefficients from a 24 hours batch
#coeffs_h_alb = [71.2910296022329, 0.0231686076492952, 0.6145542596648126, -0.1171987773974747, 1.811353871530107, 0.1727762550157405, -0.7397611689733604, -0.07067655485864967, 0.28579070256043304]
#coeffs_β_alb = [0.38443967581976596, -0.0020283723172950883, 0.031783721013116904, 0.006133949810773078, -0.038221408257357044, -0.0039048197240099964]

#Coefficients from a 36 hours batch
#coeffs_h_alb = [71.4247260603715, 0.023406648551170712, 0.5973290760963471, -0.1202824536918391, 1.7937985974512103, 0.17130826896669202, -0.7448958498839632, -0.0687836556740055, 0.29326952378678034]
#coeffs_β_alb = [0.38238060945289376, -0.002044617814571055, 0.031032187894702877, 0.006199791655980477, -0.04080964639472878, -0.0038377478577638737]
#Coefficients from a 48 hours batch
#coeffs_h_alb = [71.26228185415923, 0.02306747484712103, 0.591995542463079, -0.12262099190437198, 1.8073957756710564, 0.16969118736635064, -0.7631166804465022, -0.07007457755193583, 0.29509307713475674]
#coeffs_β_alb = [0.38891714541703926, -0.0021218729405062518, 0.03284545859871711, 0.0062636832576936285, -0.04234265195244702, -0.00408192881905754]
#Coefficients from a 60 hours batch
#coeffs_h_alb = [71.20997052773807, 0.02196608541476479, 0.6203706232089002, -0.1279504819877249, 1.800889875135111, 0.17509156726269348, -0.7259494080450594, -0.06903969459148485, 0.30890969415222186]
#coeffs_β_alb = [0.3882719741194837, -0.0022243529731177524, 0.03185107184499474, 0.006050078390818555, -0.04376119793257472, -0.004160672451438945]

p_β_alb = Polynomial(coeffs_β_alb)
p_h_alb = Polynomial(coeffs_h_alb)
pars = vcat(p_h_alb,p_β_alb)

pars_list = map(coeffs, pars)
nh = length(coeffs_h_alb)


#################################################################################################
# Targets para la inversión de background
#################################################################################################
ts_data = timestamps_test

# El target de inversión es el background inferido por el notebook de separación.
# La señal real y la perturbación se conservan solo como diagnóstico.
target_A_raw = baseline_amp[indices]
target_phi_raw = baseline_phi[indices]

A_real_used = A_real_sep[indices]
phi_real_used = phi_real_sep[indices]

A_perturbation_used = amp_detrended[indices]
phi_perturbation_used = phi_detrended[indices]

A_total_fit_used = A_total_fit_sep[indices]
phi_total_fit_used = phi_total_fit_sep[indices]

#################################################################################################
# Correcciones instrumentales aplicadas explícitamente al target usado por la loss
#################################################################################################
const AMP_OFFSET_DB_USED = 19.0
const PHI_OFFSET_DEG_USED = -210.0

target_A_used = target_A_raw .+ AMP_OFFSET_DB_USED
target_phi_used = target_phi_raw .+ PHI_OFFSET_DEG_USED

#################################################################################################
# Cache LMP
#################################################################################################
const CACHE = build_cache(ts_data, tx2, rx, rx_rec, ground, dists_new)

BLAS.set_num_threads(1)                       # evita oversubscribe con OpenBLAS/MKL
const NTH = max(1, Threads.nthreads()-1)

#################################################################################################
# Escalas de normalización
#################################################################################################
scale_A = sum((target_A_used .- mean(target_A_used)).^2) + 1e-12
scale_phi = sum((target_phi_used .- mean(target_phi_used)).^2) + 1e-12

const W_A = 1.0
const W_PHI = 1.0
println("Loss weights: W_A = ", W_A, ", W_PHI = ", W_PHI)

function normalized_loss_components_from_models(A_model, phi_model, target_A_used, target_phi_used, scale_A, scale_phi)
    error_A = sum((A_model .- target_A_used).^2)
    error_phi = sum((phi_model .- target_phi_used).^2)

    χ2_A = error_A / scale_A
    χ2_phi = error_phi / scale_phi
    χ2_unweighted = χ2_A + χ2_phi
    χ2_weighted = W_A * χ2_A + W_PHI * χ2_phi

    if !isfinite(χ2_weighted)
        return (χ2_A = 1e99, χ2_phi = 1e99, χ2_unweighted = 1e99, χ2_weighted = 1e99)
    end

    return (
        χ2_A = χ2_A,
        χ2_phi = χ2_phi,
        χ2_unweighted = χ2_unweighted,
        χ2_weighted = χ2_weighted,
    )
end

function normalized_loss_from_models(A_model, phi_model, target_A_used, target_phi_used, scale_A, scale_phi)
    comps = normalized_loss_components_from_models(A_model, phi_model, target_A_used, target_phi_used, scale_A, scale_phi)
    return comps.χ2_weighted
end

##############################################
# LOSS normalizada y robusta para background
##############################################
@inline function loss_bg(pp_flat, mat_sza, target_A_used, target_phi_used, cache::SimCache)
    if !all(isfinite, pp_flat)
        return 1e99
    end

    pp = [collect(Float64.(pp_flat[1:nh])), collect(Float64.(pp_flat[nh+1:end]))]

    Aₜ, ϕₜ = try
        vlfsignal_simple_perturbation_v2(zdt_ref, phi_firi_wrapped_ref, mat_sza, pp, cache)
    catch
        return 1e99
    end

    if any(!isfinite, Aₜ) || any(!isfinite, ϕₜ)
        return 1e99
    end

    error_A = sum((Aₜ .- target_A_used).^2)
    error_phi = sum((ϕₜ .- target_phi_used).^2)

    χ2_A = error_A / scale_A
    χ2_phi = error_phi / scale_phi
    χ2_unweighted = χ2_A + χ2_phi
    χ2_weighted = W_A * χ2_A + W_PHI * χ2_phi

    println(
        "Norm. error_A: ", χ2_A,
        ", norm. error_ϕ: ", χ2_phi,
        ", unweighted χ2: ", χ2_unweighted,
        ", weighted χ2: ", χ2_weighted,
    )

    return isfinite(χ2_weighted) ? χ2_weighted : 1e99
end

loss_wrapper(pp) = loss_bg(pp, mat_sza, target_A_used, target_phi_used, CACHE)

#################################################################################################
# Configuración de la inversión
#################################################################################################
const MAXTIME_SEC = 10.0 * 60 * 60

const INITIAL_SCALE = 1.0
const SEARCH_LOW_FACTOR = 0.92
const SEARCH_HIGH_FACTOR = 1.08
#const PREVIOUS_RESULTS_H5 = "new-real-background-inversion-bbo_from_old_asls_results_13256787.h5"
const PREVIOUS_RESULTS_H5 = "new-real-background-inversion-bbo_from_old_asls_restart_results_13341497.h5"

# Semilla FIRI conservada solo como referencia interna de dimensión.
pars0_firi = vcat(pars_list[1].*1.0, pars_list[2].*1.0)

# Restart controlado:
#   semilla = reconstrucción BBO del job 13256787
#   bounds  = ±8% alrededor de esa nueva semilla
prev_h_coeffs = collect(Float64.(h5read(PREVIOUS_RESULTS_H5, "reconstruction/h_coeffs")))
prev_beta_coeffs = collect(Float64.(h5read(PREVIOUS_RESULTS_H5, "reconstruction/beta_coeffs")))

if length(prev_h_coeffs) != length(pars_list[1])
    error("El número de coeficientes h del HDF5 previo no coincide con el script actual.")
end

if length(prev_beta_coeffs) != length(pars_list[2])
    error("El número de coeficientes beta del HDF5 previo no coincide con el script actual.")
end

seed_h_coeffs = prev_h_coeffs
seed_beta_coeffs = prev_beta_coeffs

initial_params = INITIAL_SCALE .* vcat(seed_h_coeffs, seed_beta_coeffs)

lower = SEARCH_LOW_FACTOR .* initial_params
upper = SEARCH_HIGH_FACTOR .* initial_params

println("Restart seed source: ", PREVIOUS_RESULTS_H5)
println("Search bounds source: ±8% around previous BBO-ASLS reconstruction")

# global_logger(NullLogger())

search_range = [(min(lower[i], upper[i]), max(lower[i], upper[i])) for i in 1:length(upper)]
num_params = length(initial_params)
nthreads = max(1, Threads.nthreads()-1)  # evita 0

#################################################################################################
# Evaluar semilla de restart antes de BBO
#################################################################################################
pp_seed = [initial_params[1:nh], initial_params[nh+1:end]]
A_seed_model, phi_seed_model = vlfsignal_simple_perturbation_v2(
    zdt_ref,
    phi_firi_wrapped_ref,
    mat_sza,
    pp_seed,
    CACHE,
)

seed_components = normalized_loss_components_from_models(
    A_seed_model,
    phi_seed_model,
    target_A_used,
    target_phi_used,
    scale_A,
    scale_phi,
)
seed_loss = seed_components.χ2_weighted

println("weighted chi2(initial restart seed) = ", seed_loss)
println("unweighted chi2(initial restart seed) = ", seed_components.χ2_unweighted)
println("seed χ2_A = ", seed_components.χ2_A, ", seed χ2_phi = ", seed_components.χ2_phi)

#################################################################################################
# BlackBoxOptim
#################################################################################################
t_bbo_start = time()

result = BlackBoxOptim.bboptimize(
    loss_wrapper,
    initial_params;
    Method = :dxnes,
    SearchRange = search_range,
    NumDimensions = num_params,
    PopulationSize = 20,
    MaxFuncEvals = 1e9,
    TraceMode = :compact,
    MaxTime = MAXTIME_SEC,
    NThreads = nthreads,
)

elapsed_time_sec = time() - t_bbo_start

println("Mejor solución encontrada: ", best_candidate(result))
println("Mejor error encontrado: ", best_fitness(result))
println("Parámetros iniciales: ", initial_params)

best_params = collect(Float64.(best_candidate(result)))
pp_reco = [best_params[1:nh], best_params[nh+1:end]]
reco_params = best_params

println("Parámetros óptimos: ", pp_reco)

A_reco_model, phi_reco_model = vlfsignal_simple_perturbation_v2(
    zdt_ref,
    phi_firi_wrapped_ref,
    mat_sza,
    pp_reco,
    CACHE,
)

reco_components = normalized_loss_components_from_models(
    A_reco_model,
    phi_reco_model,
    target_A_used,
    target_phi_used,
    scale_A,
    scale_phi,
)
reco_loss = reco_components.χ2_weighted

χ2_init = seed_loss
χ2_best = reco_loss

println("weighted chi2(init) = ", χ2_init)
println("weighted chi2(best) = ", χ2_best)
println("unweighted chi2(init) = ", seed_components.χ2_unweighted)
println("unweighted chi2(best) = ", reco_components.χ2_unweighted)
println("best χ2_A = ", reco_components.χ2_A, ", best χ2_phi = ", reco_components.χ2_phi)
println("best_fitness(BBO) = ", best_fitness(result))


#################################################################################################
# Guardado limpio: semilla de restart + reconstrucción BBO
#################################################################################################
function write_string_vector(parent, name, values)
    write(parent, name, String.(values))
end

function save_real_background_inversion_h5(
    output_h5;

    source_separation_h5,
    timestamps_iso_used,
    timestamps_unix_used,
    indices_used,

    target_A_raw,
    target_phi_raw,
    target_A_used,
    target_phi_used,

    A_real_used,
    phi_real_used,
    A_perturbation_used,
    phi_perturbation_used,
    A_total_fit_used,
    phi_total_fit_used,

    seed_h_coeffs,
    seed_beta_coeffs,
    seed_params,
    A_seed_model,
    phi_seed_model,
    seed_loss,
    seed_chi2_A,
    seed_chi2_phi,
    seed_chi2_unweighted,
    seed_chi2_weighted,

    reco_h_coeffs,
    reco_beta_coeffs,
    reco_params,
    A_reco_model,
    phi_reco_model,
    reco_loss,
    reco_chi2_A,
    reco_chi2_phi,
    reco_chi2_unweighted,
    reco_chi2_weighted,

    initial_params,
    search_lower,
    search_upper,
    best_candidate,
    best_fitness_bbo,

    parameter_names,
    bbo_method,
    max_time,
    elapsed_time_sec,

    amp_offset_db_used,
    phi_offset_deg_used,
    initial_scale,
    search_low_factor,
    search_high_factor,
    weight_A,
    weight_phi,
    loss_type = "normalized_chi2_A_phi",
    target_source = "baseline_asls_amp, baseline_asls_phi",
)
    h5open(output_h5, "w") do f
        # ----------------------------------------------------
        # Global metadata
        # ----------------------------------------------------
        HDF5.attributes(f)["method"] = "real_background_inversion_from_separation_h5"
        HDF5.attributes(f)["source_separation_h5"] = String(source_separation_h5)
        HDF5.attributes(f)["target_source"] = target_source
        HDF5.attributes(f)["loss_type"] = loss_type
        HDF5.attributes(f)["weight_A"] = Float64(weight_A)
        HDF5.attributes(f)["weight_phi"] = Float64(weight_phi)
        HDF5.attributes(f)["bbo_method"] = String(bbo_method)
        HDF5.attributes(f)["amp_offset_db_used"] = amp_offset_db_used
        HDF5.attributes(f)["phi_offset_deg_used"] = phi_offset_deg_used
        HDF5.attributes(f)["initial_scale"] = initial_scale
        HDF5.attributes(f)["search_low_factor"] = search_low_factor
        HDF5.attributes(f)["search_high_factor"] = search_high_factor
        HDF5.attributes(f)["initial_source"] = "previous_BBO_ASLS_restart_reconstruction_13341497"#"previous_BBO_ASLS_reconstruction_13256787"
        HDF5.attributes(f)["previous_results_h5"] = String(PREVIOUS_RESULTS_H5)
        HDF5.attributes(f)["bounds_source"] = "search_low_factor/search_high_factor around previous BBO-ASLS reconstruction"
        HDF5.attributes(f)["created_at"] = string(now())
        HDF5.attributes(f)["parameter_names"] = join(parameter_names, ",")

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------
        gtime = create_group(f, "time")
        write_string_vector(gtime, "timestamps_iso_used", timestamps_iso_used)
        write(gtime, "timestamps_unix_used", Float64.(timestamps_unix_used))
        write(gtime, "indices_used", Int.(indices_used))

        # ----------------------------------------------------
        # Targets
        # ----------------------------------------------------
        gt = create_group(f, "targets")
        write(gt, "target_A_raw", Float64.(target_A_raw))
        write(gt, "target_phi_raw", Float64.(target_phi_raw))
        write(gt, "target_A_used", Float64.(target_A_used))
        write(gt, "target_phi_used", Float64.(target_phi_used))
        HDF5.attributes(gt)["description"] = (
            "target_*_raw comes from the separation HDF5. " *
            "target_*_used is the corrected version used in the normalized loss."
        )

        # ----------------------------------------------------
        # FIRI seed
        # ----------------------------------------------------
        gs = create_group(f, "seed")
        write(gs, "h_coeffs", Float64.(seed_h_coeffs))
        write(gs, "beta_coeffs", Float64.(seed_beta_coeffs))
        write(gs, "params_vector", Float64.(seed_params))
        write(gs, "A_model", Float64.(A_seed_model))
        write(gs, "phi_model", Float64.(phi_seed_model))
        HDF5.attributes(gs)["loss"] = Float64(seed_loss)
        HDF5.attributes(gs)["weighted_loss"] = Float64(seed_chi2_weighted)
        HDF5.attributes(gs)["unweighted_loss"] = Float64(seed_chi2_unweighted)
        HDF5.attributes(gs)["chi2_A"] = Float64(seed_chi2_A)
        HDF5.attributes(gs)["chi2_phi"] = Float64(seed_chi2_phi)
        HDF5.attributes(gs)["description"] = "Forward model evaluated at the initial restart seed coefficients loaded from the previous ASLS inversion."

        # ----------------------------------------------------
        # Reconstructed solution
        # ----------------------------------------------------
        gr = create_group(f, "reconstruction")
        write(gr, "h_coeffs", Float64.(reco_h_coeffs))
        write(gr, "beta_coeffs", Float64.(reco_beta_coeffs))
        write(gr, "params_vector", Float64.(reco_params))
        write(gr, "A_model", Float64.(A_reco_model))
        write(gr, "phi_model", Float64.(phi_reco_model))
        HDF5.attributes(gr)["loss"] = Float64(reco_loss)
        HDF5.attributes(gr)["weighted_loss"] = Float64(reco_chi2_weighted)
        HDF5.attributes(gr)["unweighted_loss"] = Float64(reco_chi2_unweighted)
        HDF5.attributes(gr)["chi2_A"] = Float64(reco_chi2_A)
        HDF5.attributes(gr)["chi2_phi"] = Float64(reco_chi2_phi)
        HDF5.attributes(gr)["description"] = "Best solution returned by BlackBoxOptim, evaluated with the forward model."

        # ----------------------------------------------------
        # Optimization metadata
        # ----------------------------------------------------
        go = create_group(f, "optimization")
        write(go, "initial_params", Float64.(initial_params))
        write(go, "search_lower", Float64.(search_lower))
        write(go, "search_upper", Float64.(search_upper))
        write(go, "best_candidate", Float64.(best_candidate))
        HDF5.attributes(go)["best_fitness_bbo"] = Float64(best_fitness_bbo)
        HDF5.attributes(go)["best_fitness_is_weighted_loss"] = true
        HDF5.attributes(go)["weight_A"] = Float64(weight_A)
        HDF5.attributes(go)["weight_phi"] = Float64(weight_phi)
        HDF5.attributes(go)["max_time"] = Float64(max_time)
        HDF5.attributes(go)["elapsed_time_sec"] = Float64(elapsed_time_sec)
        HDF5.attributes(go)["method"] = String(bbo_method)

        # ----------------------------------------------------
        # Diagnostics from the separation file.
        # These arrays are sampled at the inversion times but are not used in the loss.
        # ----------------------------------------------------
        gd = create_group(f, "diagnostics")
        write(gd, "A_real_used", Float64.(A_real_used))
        write(gd, "phi_real_used", Float64.(phi_real_used))
        write(gd, "A_perturbation_used", Float64.(A_perturbation_used))
        write(gd, "phi_perturbation_used", Float64.(phi_perturbation_used))
        write(gd, "A_total_fit_used", Float64.(A_total_fit_used))
        write(gd, "phi_total_fit_used", Float64.(phi_total_fit_used))
        HDF5.attributes(gd)["description"] = (
            "Diagnostic arrays from the separation file at the inversion times. " *
            "They are not used in the background inversion loss."
        )
    end

    println("Inversion results saved to: ", output_h5)
end

jobid = get(ENV, "SLURM_JOB_ID", "nojid")
filename_hdf5 = "new-real-background-inversion-bbo_from_old_asls_restart_results_$(jobid).h5"

parameter_names = vcat(
    ["h_c$(i-1)" for i in 1:length(pars_list[1])],
    ["beta_c$(i-1)" for i in 1:length(pars_list[2])],
)

timestamps_iso_used = string.(zdt_test)

save_real_background_inversion_h5(
    filename_hdf5;

    source_separation_h5 = filename_sep,
    timestamps_iso_used = timestamps_iso_used,
    timestamps_unix_used = timestamps_test,
    indices_used = indices,

    target_A_raw = target_A_raw,
    target_phi_raw = target_phi_raw,
    target_A_used = target_A_used,
    target_phi_used = target_phi_used,

    A_real_used = A_real_used,
    phi_real_used = phi_real_used,
    A_perturbation_used = A_perturbation_used,
    phi_perturbation_used = phi_perturbation_used,
    A_total_fit_used = A_total_fit_used,
    phi_total_fit_used = phi_total_fit_used,

    seed_h_coeffs = seed_h_coeffs,
    seed_beta_coeffs = seed_beta_coeffs,
    seed_params = initial_params,
    A_seed_model = A_seed_model,
    phi_seed_model = phi_seed_model,
    seed_loss = seed_loss,
    seed_chi2_A = seed_components.χ2_A,
    seed_chi2_phi = seed_components.χ2_phi,
    seed_chi2_unweighted = seed_components.χ2_unweighted,
    seed_chi2_weighted = seed_components.χ2_weighted,

    reco_h_coeffs = best_params[1:nh],
    reco_beta_coeffs = best_params[nh+1:end],
    reco_params = reco_params,
    A_reco_model = A_reco_model,
    phi_reco_model = phi_reco_model,
    reco_loss = reco_loss,
    reco_chi2_A = reco_components.χ2_A,
    reco_chi2_phi = reco_components.χ2_phi,
    reco_chi2_unweighted = reco_components.χ2_unweighted,
    reco_chi2_weighted = reco_components.χ2_weighted,

    initial_params = initial_params,
    search_lower = lower,
    search_upper = upper,
    best_candidate = best_params,
    best_fitness_bbo = best_fitness(result),

    parameter_names = parameter_names,
    bbo_method = :dxnes,
    max_time = MAXTIME_SEC,
    elapsed_time_sec = elapsed_time_sec,

    amp_offset_db_used = AMP_OFFSET_DB_USED,
    phi_offset_deg_used = PHI_OFFSET_DEG_USED,
    initial_scale = INITIAL_SCALE,
    search_low_factor = SEARCH_LOW_FACTOR,
    search_high_factor = SEARCH_HIGH_FACTOR,
    weight_A = W_A,
    weight_phi = W_PHI,
    loss_type = "normalized_chi2_A_phi",
    target_source = "baseline_asls_amp, baseline_asls_phi",
)

