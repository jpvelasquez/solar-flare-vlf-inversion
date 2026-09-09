#!/usr/bin/env julia

################################################################################
# Whittaker-background inversion: heterogeneous lower boundary — BBO-2 restart
#
# Ground configuration, ordered from NAA (TX) to PLO (RX):
#   [GROUND[10] × 5, GROUND[7] × 5]
#
# This is the second heterogeneous inversion:
#   - seed = valid heterogeneous BBO-1 reconstruction from run 16592871;
#   - target remains unchanged: A_background + 19 dB and phi_background - 210 deg;
#   - search bounds = ±15% around the BBO-1 reconstruction;
#   - same normalized objective, 10-waveguide geometry, and 18:00–22:00 UTC sampling;
#   - same heterogeneous ground and same-ground FIRI phase reference;
#   - no coefficients from the offset-modified diagnostic runs are used.
################################################################################

const ROOT_DIR = @__DIR__
cd(ROOT_DIR)

#####################################################################################
#################### Loading Julia files needed ##  ###################################
#####################################################################################
################# preamble.jl contains all the packages necessary in this simulation.
include(joinpath(ROOT_DIR, "mytools", "preamble.jl"))
##########################################################################################################################################
######### scenario.jl contains information on TX and RX position,
######### their features (VerticalDipole and GroundSampler respectively),
######### geomagnetic field, altitude, initial and final time, etc. ######################################################################################
include(joinpath(ROOT_DIR, "mytools", "scenario.jl"))

const GROUND_CASE_KEY = "ocean5_goodsoil5"
const GROUND_INDICES_TX_TO_RX = vcat(fill(10, 5), fill(7, 5))
const GROUND_BY_SEGMENT = Any[GROUND[i] for i in GROUND_INDICES_TX_TO_RX]

length(GROUND_INDICES_TX_TO_RX) == 10 || error(
    "The heterogeneous configuration must contain exactly 10 segments."
)
all(GROUND_INDICES_TX_TO_RX[1:5] .== 10) || error(
    "Segments 1:5 must use GROUND[10] (sea water)."
)
all(GROUND_INDICES_TX_TO_RX[6:10] .== 7) || error(
    "Segments 6:10 must use GROUND[7] (good soil)."
)

const GROUND_EPSILON_R_BY_SEGMENT = Float64[
    GROUND[i].ϵᵣ for i in GROUND_INDICES_TX_TO_RX
]
const GROUND_SIGMA_S_M_BY_SEGMENT = Float64[
    GROUND[i].σ for i in GROUND_INDICES_TX_TO_RX
]

println("="^88)
println("Whittaker-background heterogeneous inversion — BBO-2 restart15")
println("Ground case: ", GROUND_CASE_KEY)
println("GROUND indices, TX -> RX: ", GROUND_INDICES_TX_TO_RX)
println("epsilon_r by segment: ", GROUND_EPSILON_R_BY_SEGMENT)
println("sigma [S/m] by segment: ", GROUND_SIGMA_S_M_BY_SEGMENT)
println("="^88)
import CairoMakie
const CM = CairoMakie
using Optim
using LinearAlgebra  
using Polynomials
using GMT
using Dates
#using SavitzkyGolay, Statistics
##################################################################
using Interpolations: interpolate, extrapolate, Gridded, Linear, Line
using SavitzkyGolay
using AstroLib
using DSP
using JLD2
##################################################################
#unique(FIRI.HEADER.f10_7)
#FIRI.HEADER[1960:1980]
#ALT_KM = FIRI.ALTITUDE/1e3
###############################################################3
### Funciones necesarias para el análisis
################################################################
##################################################################

function PlotTrajectories()
    
    output_file = "/home/jp/basefolder_JP/vlf-trajectories"
    lon_min, lon_max = -120, -60  # Ejemplo: América del Norte
    lat_min, lat_max = -10, 50     # Ejemplo: Norteamérica y el Caribe
    # Configuración del gráfico de la costa
    mid_point2 = [sum([lon_min,lon_max])/2, sum([lat_min,lat_max])/2]
    GMT.coast(region=:global, xaxis=(annot=60,), yaxis=(annot=30,), proj=(name=:ortho, center=mid_point), 
    frame=:g, water=:lightblue, land=:darkgreen,
    title="$tx_station2 - $rx_station / $rx_station2 propagation path")#,savefig="$output_file.pdf")
    # Primera trayectoria
    GMT.plot!(lons, lats, lw=1, lc=:navy, marker=:circle, size=0.2, 
          markeredgecolor=0, markerfacecolor=:navy)
    # Segunda trayectoria
    GMT.plot!(lons2, lats2, lw=1, lc=:red, marker=:circle, size=0.2, 
              markeredgecolor=0, markerfacecolor=:red)
    # Agregar etiqueta "PIU" al final de la primera trayectoria
    GMT.text!([lons[1] lats[1]], text="WWVB (TX)", font=(10, :black), justify=:RB)
    GMT.text!([lons[end] lats[end]], text="PLO", font=(10, :navy), justify=:LB)
    # Agregar etiqueta "PLO" al final de la segunda trayectoria
    GMT.text!([lons2[end] lats2[end]], text="PIU", font=(10, :red), justify=:LB, show=true)
    end
##################################################################
function GetParams(initial_guess, firi_prof,alt_aux)
    ff(x) = obj_func(x[1],x[2],firi_prof,alt_aux)
    options = Optim.Options(iterations=10000)
    result = optimize(ff,initial_guess,options)
    h_reco = Optim.minimizer(result)[1]
    β_reco = Optim.minimizer(result)[2]
    reco_prof = waitprofile.(alt_aux,h_reco,β_reco)
    chi2_value = sum((reco_prof .- firi_prof).^ 2)
    return h_reco, β_reco, chi2_value 
end
#####################################################################
function reflectionheight(ne_firi_prof)
    ne_xz_prof =  Interpolations.interpolate(alt, ne_firi_prof,
                    FritschButlandMonotonicInterpolation());
    species = Species(QE, ME,ne_xz_prof, electroncollisionfrequency)
    ωr = waitsparameter.(alt, (tx.frequency,), (bfield,), (species,))
    itp = linear_interpolation(ωr, alt)
    eqz = itp(tx.frequency.ω)
    #println("eqz = ",eqz)
    return eqz
end
####################################################################
function reflectionheight_v2(   ne_prof::AbstractVector{<:Real},
                                tol::Real=1e-2,
                                maxiter::Int=80)::Float64
    # interpolador/extrapolador de ne(h)
    tx_freq = tx.frequency#.ω
    itp0   = interpolate((alt,), ne_prof, Gridded(Linear()))
    ne_itp = extrapolate(itp0, Line())
    sp     = Species(QE, ME, ne_itp, electroncollisionfrequency)
    ω_req  = tx_freq.ω
    # función a raíz: ωr(h) – ω_req
    f(h) = waitsparameter(h, tx_freq, bfield, sp) - ω_req
    a, b = alt[1]+100.0, alt[end]-100.0
    fa, fb = f(a), f(b)
    if fa*fb > 0
        return NaN
    end
    # bisección
    for _ in 1:maxiter
        c, fc = 0.5*(a+b), f(0.5*(a+b))
        if abs(fc) < tol
            return c/1e3
        elseif fa*fc < 0
            b, fb = c, fc
        else
            a, fa = c, fc
        end
    end
    return NaN
end


##################################################################
function obj_func(h,β,firi_prof,alt_aux)
    wait = waitprofile.(alt_aux,h,β)
    diff = norm(firi_prof.-wait)
    return diff
end
###################################################################
function GetIndices(ne_firi,refHeights)
    hs_round = Float64.(round.(refHeights/1e3))
    inds = []
    for i in 1:size(ne_firi)[2]
        index = findfirst(==(hs_round[i]), z)
        push!(inds,index)
    end
    return inds
end
###################################################################
function FitPolinomial(x,y,n)
    p = fit(x, y, n)
    #x_eval = dists#1:0.1:10      # Valores para evaluar el polinomio
    y_eval = p.(x)    # Evaluar el polinomio
    chi2_h_fit = Float64(sum((y_eval.-y).^2 ./y ))
    #chi2_h_fit = Float64(sum((y_eval.-y).^2 ))
    return p, chi2_h_fit
end
###################################################################
function FitDifferentDegrees(dists,sza,hs_reco,βs_reco)
    x = Float64.(dists)
    x_sz = Float64.(sza)
    y_h = Float64.(hs_reco)
    y_β = Float64.(βs_reco)
    y_h_sz = Float64.(hs_reco)
    y_β_sz = Float64.(βs_reco)
    chi2_h_dist = []
    chi2_β_dist = []
    chi2_h_sza = []
    chi2_β_sza = []
    n_order = 1:1:18
    for n in n_order
        p_h, chi2_h_fit = FitPolinomial(x,y_h,n)
        p_β, chi2_β_fit = FitPolinomial(x,y_β,n)
        p_h_sz, chi2_h_fit_sz = FitPolinomial(x_sz,y_h_sz,n)
        p_β_sz, chi2_β_fit_sz = FitPolinomial(x_sz,y_β_sz,n)
        h_eval = p_h.(x)
        β_eval = p_β.(x)
        h_eval_sz = p_h_sz.(x_sz)
        β_eval_sz = p_β_sz.(x_sz)
    
        push!(chi2_h_dist,chi2_h_fit)
        push!(chi2_β_dist,chi2_β_fit)
        push!(chi2_h_sza,chi2_h_fit_sz)
        push!(chi2_β_sza,chi2_β_fit_sz)
    end
    return chi2_h_dist, chi2_β_dist, chi2_h_sza, chi2_β_sza
end
###################################################################
function FitDifferentDegrees_v2(sza,hs_reco,βs_reco)
    x = Float64.(sza)
    y_h = Float64.(hs_reco)
    y_β = Float64.(βs_reco)
    
    chi2_h_sza = []
    chi2_β_sza = []
    n_order = 1:1:18
    for n in n_order
        p_h, chi2_h_fit = FitPolinomial(x,y_h,n)
        p_β, chi2_β_fit = FitPolinomial(x,y_β,n)
        h_eval = p_h.(x)
        β_eval = p_β.(x)
        push!(chi2_h_sza,chi2_h_fit)
        push!(chi2_β_sza,chi2_β_fit)
    end

    return chi2_h_sza, chi2_β_sza
end
#####################################################################
#####################################################################
function GetAzimuthPolynomials_v1(zdt_test,latstep=1,nh=3,nβ=4)
    hmin = 50#70#73.0
    hmax = 90#69.5#77.5
    hstep = 0.5 
    hrefs_temp = hmin:hstep:hmax
    initial_guess = [75.0,0.3]
    hs_all = []
    βs_all = []
    sza_all = []
    for tt in zdt_test
        str_aux_temp = Dates.format(tt, "HH:MM")*" UT"
        lats_temp = lats[1:latstep:end]
        lons_temp = lons[1:latstep:end]
        sza_temp = get_sza.(lats_temp, lons_temp, tt)
        ne_firi_temp = nefiri(lat=lats_temp, lon=lons_temp, zdt=tt, findex=75)
        refHeights_temp = []
        for i in 1:1:size(ne_firi_temp)[2]
            fprof = ne_firi_temp[:,i]
            eqz = reflectionheight_v2(fprof)*1e3
            push!(refHeights_temp,eqz)
        end
        inds = GetIndices(ne_firi_temp,refHeights_temp)
        hs_round = Float64.(round.(refHeights_temp))
        hs_reco = []
        βs_reco = []
        chi2_values = []
        ind_inf = 1
        for i in 1:1:size(ne_firi_temp)[2]
            index = inds[i]
            fprof = ne_firi_temp[:,i][ind_inf:index]
            alt_aux = alt[ind_inf:index]
            h_reco, β_reco, chi2_value = GetParams(initial_guess, fprof,alt_aux)
            reco_prof = waitprofile.(alt_aux,h_reco,β_reco)
            push!(hs_reco,h_reco)
            push!(βs_reco,β_reco)
            push!(chi2_values,chi2_value)
            push!(hs_all,h_reco)
            push!(βs_all,β_reco)
            push!(sza_all,sza_temp[i])
            initial_guess = [h_reco, β_reco]
        end
    #####################################################################
    end
    sza_all = Float64.(sza_all)
    hs_all = Float64.(hs_all)
    βs_all = Float64.(βs_all)
    println(length(sza_all))
    sza_aux = 0.0:0.5:(round(maximum(sza_all))+1)
    sza_all_rad = sza_all.*(pi/180.0)
    sza_aux_rad = sza_aux.*(pi/180.0)
    p_h_sza, chi_h_sza_fit = FitPolinomial(sza_all_rad,hs_all,nh)
    p_β_sza, chi_β_sza_fit = FitPolinomial(sza_all_rad,βs_all,nβ)
    
    return p_h_sza, chi_h_sza_fit, p_β_sza, chi_β_sza_fit
end
#####################################################################
function GetAzimuthPolynomials(zdt_test,latstep=1)
    hmin = 50#70#73.0
    hmax = 90#69.5#77.5
    hstep = 0.5 
    hrefs_temp = hmin:hstep:hmax
    initial_guess = [78.0,0.3]
    hs_all = []
    βs_all = []
    sza_all = []
    for tt in zdt_test
        str_aux_temp = Dates.format(tt, "HH:MM")*" UT"
        lats_temp = lats[1:latstep:end]
        lons_temp = lons[1:latstep:end]
        sza_temp = get_sza.(lats_temp, lons_temp, tt)
        #ne_firi_temp = nefiri(lat=lats_temp, lon=lons_temp, zdt=tt, findex=75)
        ne_firi_temp = nefiri(lat=lats_temp, lon=lons_temp, zdt=tt, findex=75)
        #ne_firi_temp = SG_Filter(ne_firi_temp0,win,ord)
        refHeights_temp = []
        for i in 1:1:size(ne_firi_temp)[2]
            fprof = ne_firi_temp[:,i]
            eqz = reflectionheight_v2(fprof)*1e3
            push!(refHeights_temp,eqz)
        end
        inds = GetIndices(ne_firi_temp,refHeights_temp)
        hs_round = Float64.(round.(refHeights_temp))
        hs_reco = []
        βs_reco = []
        chi2_values = []
        ind_inf = 1
        for i in 1:1:size(ne_firi_temp)[2]
            index = inds[i]
            fprof = ne_firi_temp[:,i][ind_inf:index]
            alt_aux = alt[ind_inf:index]
            h_reco, β_reco, chi2_value = GetParams(initial_guess, fprof,alt_aux)
            reco_prof = waitprofile.(alt_aux,h_reco,β_reco)
            push!(hs_reco,h_reco)
            push!(βs_reco,β_reco)
            push!(chi2_values,chi2_value)
            push!(hs_all,h_reco)
            push!(βs_all,β_reco)
            push!(sza_all,sza_temp[i])
            initial_guess = [h_reco, β_reco]
        end
    #####################################################################
    end
    sza_all = Float64.(sza_all)
    hs_all = Float64.(hs_all)
    βs_all = Float64.(βs_all)
    println(length(sza_all))
    #sza_aux = 0.0:0.5:103.0
    sza_aux = 0.0:0.5:(round(maximum(sza_all))+1)
    sza_all_rad = sza_all.*(pi/180.0)
    sza_aux_rad = sza_aux.*(pi/180.0)
    #println(sza_aux[end])
    p_h_sza, chi_h_sza_fit = FitPolinomial(sza_all_rad,hs_all,6) #7
    p_β_sza, chi_β_sza_fit = FitPolinomial(sza_all_rad,βs_all,6) #7
    
    return p_h_sza, chi_h_sza_fit, p_β_sza, chi_β_sza_fit
end
#####################################################################
#####################################################################
function GetAzimuthPolynomials_filt(zdt_test,latstep=1)
    hmin = 50#70#73.0
    hmax = 90#69.5#77.5
    hstep = 0.5 
    hrefs_temp = hmin:hstep:hmax
    initial_guess = [75.0,0.3]
    hs_all = []
    βs_all = []
    sza_all = []
    for tt in zdt_test
        str_aux_temp = Dates.format(tt, "HH:MM")*" UT"
        lats_temp = lats[1:latstep:end]
        lons_temp = lons[1:latstep:end]
        sza_temp = get_sza.(lats_temp, lons_temp, tt)
        ne_firi_temp0 = nefiri(lat=lats_temp, lon=lons_temp, zdt=tt, findex=75)
        ne_firi_temp = SG_Filter(ne_firi_temp0,win,ord)
        refHeights_temp = []
        for i in 1:1:size(ne_firi_temp)[2]
            fprof = ne_firi_temp[:,i]
            eqz = reflectionheight_v2(fprof)*1e3
            push!(refHeights_temp,eqz)
        end
        inds = GetIndices(ne_firi_temp,refHeights_temp)
        hs_round = Float64.(round.(refHeights_temp))
        hs_reco = []
        βs_reco = []
        chi2_values = []
        ind_inf = 1
        for i in 1:1:size(ne_firi_temp)[2]
            index = inds[i]
            fprof = ne_firi_temp[:,i][ind_inf:index]
            alt_aux = alt[ind_inf:index]
            h_reco, β_reco, chi2_value = GetParams(initial_guess, fprof,alt_aux)
            #reco_prof = waitprofile.(alt_aux,h_reco,β_reco)
            push!(hs_reco,h_reco)
            push!(βs_reco,β_reco)
            push!(chi2_values,chi2_value)
            push!(hs_all,h_reco)
            push!(βs_all,β_reco)
            push!(sza_all,sza_temp[i])
            initial_guess = [h_reco, β_reco]
        end
    #####################################################################
    end
    sza_all = Float64.(sza_all)
    hs_all = Float64.(hs_all)
    βs_all = Float64.(βs_all)
    println(length(sza_all))
    #sza_aux = 0.0:0.5:103.0
    sza_aux = 0.0:0.5:(round(maximum(sza_all))+1)
    sza_all_rad = sza_all.*(pi/180.0)
    sza_aux_rad = sza_aux.*(pi/180.0)
    #println(sza_aux[end])
    p_h_sza, chi_h_sza_fit = FitPolinomial(sza_all_rad,hs_all,6)
    p_β_sza, chi_β_sza_fit = FitPolinomial(sza_all_rad,βs_all,6)
    
    return p_h_sza, chi_h_sza_fit, p_β_sza, chi_β_sza_fit
end
#####################################################################
function SG_Filter(ne_aux, win, ord)
    sm_rows = similar(ne_aux)
    sm_2d    = similar(ne_aux)
    
    # 1) Suavizado por filas
    for i in axes(ne_aux, 1)
        sg = savitzky_golay(ne_aux[i, :], win, ord)
        sm_rows[i, :] .= sg.y
    end
    
    # 2) Suavizado por columnas del resultado anterior
    for j in axes(ne_aux, 2)
        sg = savitzky_golay(sm_rows[:, j], win, ord)
        sm_2d[:, j]    .= sg.y
    end  
    return sm_2d
end
# 0) PREP: cache de constantes
###############################
Base.@kwdef struct SimCache
    tx::Any
    rx::Any
    rx_rec::Any
    grounds::Vector{Any}
    dists12::Vector{Float64}   # dists_new[1:2:end]
    dists22::Vector{Float64}   # dists_new[2:2:end]
    dt::Vector{ZonedDateTime}  # ZonedDateTime.(unix2datetime.(ts), tz"UTC")
    bf_list::Vector{Any}       # campo magnético precomputado por tiempo
end

function build_cache(ts, tx, rx, rx_rec, grounds, dists_new)
    dt = ZonedDateTime.(unix2datetime.(ts), tz"UTC")
    dists = Float64.(dists_new)
    grounds_vec = Any[grounds...]

    # geolinspace("NAA", "PLO", 19, false) returns 21 effective points:
    #
    #   segment starts: 1, 3, ..., 19  -> 10 entries
    #   midpoints:      2, 4, ..., 20  -> 10 entries
    #   final endpoint: 21             -> receiver only
    #
    # Therefore dists_new[1:2:end] has 11 values because it also contains
    # the final receiver endpoint. That endpoint must not be interpreted as
    # the start of an additional waveguide segment.
    n_path_points = length(dists)

    n_path_points == 21 || error(
        "Expected 21 effective path points for 10 waveguides, " *
        "but geolinspace returned $n_path_points."
    )

    start_indices = collect(1:2:(n_path_points - 2))
    midpoint_indices = collect(2:2:(n_path_points - 1))

    d12 = dists[start_indices]
    d22 = dists[midpoint_indices]

    length(d12) == length(d22) || error(
        "Segment-start and midpoint arrays have different lengths after " *
        "excluding the final receiver endpoint."
    )
    length(grounds_vec) == length(d12) || error(
        "The ground configuration has $(length(grounds_vec)) entries, " *
        "but the geometry has $(length(d12)) waveguide segments."
    )
    length(d12) == 10 || error(
        "Expected exactly 10 waveguide segments, found $(length(d12))."
    )

    first(d12) == 0.0 || error(
        "The first waveguide segment does not start at 0 m."
    )
    all(diff(d12) .> 0.0) || error(
        "Waveguide segment-start distances are not strictly increasing."
    )
    all(diff(d22) .> 0.0) || error(
        "Waveguide midpoint distances are not strictly increasing."
    )
    all(d12 .< d22) || error(
        "At least one segment start is not before its midpoint."
    )

    bf_list = [igrf(tx, rx_rec, year(ti), d22; alt=60e3) for ti in dt]

    println("Geometry cache:")
    println("  effective path points = ", n_path_points)
    println("  waveguide segments    = ", length(d12))
    println("  segment starts [km]   = ", round.(d12 ./ 1e3; digits=3))
    println("  midpoints [km]        = ", round.(d22 ./ 1e3; digits=3))

    return SimCache(
        tx = tx,
        rx = rx,
        rx_rec = rx_rec,
        grounds = grounds_vec,
        dists12 = d12,
        dists22 = d22,
        dt = dt,
        bf_list = bf_list,
    )
end

##############################################
# 1) TU FUNCIÓN (firma cambiada SOLO por cache)
##############################################
function vlfsignal_simple_perturbation_v1(ts, mat_sza, pars_list, cache::SimCache)
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

        length(cache.grounds) == Nwaveguide || error(
            "Ground/geometry mismatch inside the forward model."
        )

        wg2 = SegmentedWaveguide([
            HomogeneousWaveguide(
                bf_k[j],
                species2[j],
                cache.grounds[j],
                cache.dists12[j],
            )
            for j in 1:Nwaveguide
        ])

        _, A_fit_seg, ϕ_fit_seg = propagate(wg2, cache.tx, cache.rx)
        A_fitₜ[k] = Float64(A_fit_seg)
        ϕ_fitₜ[k] = Float64(ϕ_fit_seg)
    end

    ϕ_fitₜ = rad2deg.(unwrap(ϕ_fitₜ))
    return A_fitₜ, ϕ_fitₜ
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

        length(cache.grounds) == Nwaveguide || error(
            "Ground/geometry mismatch inside the forward model."
        )

        wg2 = SegmentedWaveguide([
            HomogeneousWaveguide(
                bf_k[j],
                species2[j],
                cache.grounds[j],
                cache.dists12[j],
            )
            for j in 1:Nwaveguide
        ])

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
using HDF5

# save_results_h5/load_results_h5 del flujo antiguo fueron retirados.
# El guardado activo de este script usa save_real_background_inversion_h5 más abajo.

########################################################################
using AstroLib, TimeZones

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
using Dates
using DSP
using Interpolations: LinearInterpolation
using TimeZones: astimezone, ZonedDateTime, tz

const LinBC = Interpolations.Linear()

function unwrap_resampled(
    t_ref::Vector{ZonedDateTime},
    phi_wr_ref::AbstractVector{<:Real},
    t_new::Vector{ZonedDateTime},
    phi_wr_new::AbstractVector{<:Real};
    discont = 0.9π,
)
    @assert length(t_ref) == length(phi_wr_ref)
    @assert length(t_new) == length(phi_wr_new)
    @assert issorted(t_ref)
    @assert issorted(t_new)

    # 1) Apply DSP.unwrap only after each complete time series is available.
    phi_ref_unw = DSP.unwrap(
        collect(Float64.(phi_wr_ref));
        dims = 1,
        discont = discont,
        period = 2π,
    )

    phi_new_unw = DSP.unwrap(
        collect(Float64.(phi_wr_new));
        dims = 1,
        discont = discont,
        period = 2π,
    )

    # 2) Interpolate the independently unwrapped same-ground FIRI reference
    #    to the candidate-model times.
    to_ms(zdt_vec) = Float64.(
        Dates.value.(
            DateTime.(
                astimezone.(zdt_vec, Ref(tz"UTC"))
            )
        )
    )

    x_ref = to_ms(t_ref)
    x_new = to_ms(t_new)

    itp_phi = LinearInterpolation(
        x_ref,
        phi_ref_unw;
        extrapolation_bc = LinBC,
    )
    phi_ref_at_new = itp_phi.(x_new)

    # 3) The reference selects only one global 2π branch for the complete
    #    candidate series. No point-by-point branch adjustment is allowed.
    branch_cycles = round(
        Int,
        median((phi_ref_at_new .- phi_new_unw) ./ (2π)),
    )

    return phi_new_unw .+ branch_cycles * 2π
end
#############################################################################################################################
################################   Cargando fase de referencia de FIRI para wl unwrapping  ##################################
##############################################################################################################################
const filename_ref = joinpath(
    ROOT_DIR,
    "outputs_reference_efield_firi_10waveguides_13ground_cases_20080325",
    "reference-efield-firi_ocean5_goodsoil5_10wg_20080325.jld2",
)

isfile(filename_ref) || error(
    "The heterogeneous FIRI reference file was not found:\n$filename_ref"
)

jldfile_ref = jldopen(filename_ref, "r")
A_firi_ref = Float64.(vec(jldfile_ref["amp"]))
phi_firi_wrapped_ref = Float64.(vec(jldfile_ref["phi"]))
zdt_ref = collect(jldfile_ref["zdt"])
reference_case_key = String(jldfile_ref["case_key"])
reference_ground_indices = Int.(
    vec(jldfile_ref["ground_indices_TX_to_RX"])
)
close(jldfile_ref)

reference_case_key == GROUND_CASE_KEY || error(
    "Unexpected reference case_key: $reference_case_key"
)
reference_ground_indices == GROUND_INDICES_TX_TO_RX || error(
    "The reference-file ground configuration does not match the inversion."
)
length(zdt_ref) == length(phi_firi_wrapped_ref) || error(
    "Reference time/phase length mismatch."
)
length(A_firi_ref) == length(zdt_ref) || error(
    "Reference amplitude/time length mismatch."
)
issorted(zdt_ref) || error("Reference times are not sorted.")
all(isfinite, phi_firi_wrapped_ref) || error(
    "The wrapped reference phase contains non-finite values."
)

println("Mixed-ground FIRI reference loaded:")
println("  ", filename_ref)
println("  case_key: ", reference_case_key)
println("  GROUND indices, TX -> RX: ", reference_ground_indices)
println("  time range: ", first(zdt_ref), " -> ", last(zdt_ref))
##############################################################################################################################
############################################## Cargando los datos   ##########################################################
##############################################################################################################################
using HDF5
using Dates
using TimeZones

##############################################################################################################################
#################### Cargando HDF5 Whittaker seleccionado para inversión de background #######################################
##############################################################################################################################

# Archivo generado por el notebook de separación Whittaker-only.
# Contiene el intervalo completo 16:00–22:00 UT.
# IMPORTANTE: la inversión sigue usando solamente 18:00–22:00 UT mediante zdt_test e indices.
filename_sep = joinpath(ROOT_DIR, "real_whittaker_selected_background_16_22_for_inversion.h5")

# Lectura directa: este script usa SOLO el esquema nuevo del archivo Whittaker seleccionado.
# No hay fallbacks para nombres antiguos.
timestamps_unix_sep = Float64.(vec(h5read(filename_sep, "timestamps_unix")))
time_datetime = unix2datetime.(timestamps_unix_sep)
time_zoned = ZonedDateTime.(time_datetime, tz"UTC")
timestamps_iso = string.(time_zoned)

A_real_sep = Float64.(vec(h5read(filename_sep, "A_real")))
phi_real_sep = Float64.(vec(h5read(filename_sep, "phi_real")))
A_background_sep = Float64.(vec(h5read(filename_sep, "A_background")))
phi_background_sep = Float64.(vec(h5read(filename_sep, "phi_background")))
A_perturbation_sep = Float64.(vec(h5read(filename_sep, "A_perturbation")))
phi_perturbation_sep = Float64.(vec(h5read(filename_sep, "phi_perturbation")))
A_total_fit_sep = Float64.(vec(h5read(filename_sep, "A_total_fit")))
phi_total_fit_sep = Float64.(vec(h5read(filename_sep, "phi_total_fit")))

# Compatibilidad con la lógica original del script:
# esta inversión ajusta SOLO el background.
baseline_amp = A_background_sep
baseline_phi = phi_background_sep
amp_detrended = A_perturbation_sep
phi_detrended = phi_perturbation_sep

# Verificación mínima de consistencia dimensional.
nsep = length(time_zoned)
arrays_to_check = Dict(
    "A_real" => A_real_sep,
    "phi_real" => phi_real_sep,
    "A_background" => A_background_sep,
    "phi_background" => phi_background_sep,
    "A_perturbation" => A_perturbation_sep,
    "phi_perturbation" => phi_perturbation_sep,
    "A_total_fit" => A_total_fit_sep,
    "phi_total_fit" => phi_total_fit_sep,
)

for (name, arr) in arrays_to_check
    if length(arr) != nsep
        error("Dataset $(name) tiene longitud $(length(arr)); timestamps_unix tiene longitud $(nsep).")
    end
end

println("Archivo de separación Whittaker cargado: ", filename_sep)
println("Número de muestras cargadas: ", nsep)
println("Rango temporal del archivo:")
println("  ", first(time_zoned))
println("  ", last(time_zoned))
println("Datasets usados:")
println("  timestamps_unix")
println("  A_background")
println("  phi_background")
println("  A_perturbation")
println("  phi_perturbation")
println("  A_total_fit")
println("  phi_total_fit")

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
### Semilla del restart: best válido de la BBO-1 heterogénea
###############################################################

const PREVIOUS_RESULTS_H5 = abspath(get(
    ENV,
    "HETERO_BBO1_H5",
    joinpath(
        ROOT_DIR,
        "outputs_real_background_inversion_whittaker_ocean5_goodsoil5_BBO1",
        "new-real-background-inversion_whittaker_ocean5_goodsoil5_BBO1_results_16592871.h5",
    ),
))

isfile(PREVIOUS_RESULTS_H5) || error(
    "No se encontró el HDF5 de la BBO-1 heterogénea válida:\n" *
    PREVIOUS_RESULTS_H5 *
    "\nPuede definirse otra ruta mediante la variable HETERO_BBO1_H5."
)

previous_h_coeffs = Float64.(
    vec(h5read(PREVIOUS_RESULTS_H5, "reconstruction/h_coeffs"))
)
previous_beta_coeffs = Float64.(
    vec(h5read(PREVIOUS_RESULTS_H5, "reconstruction/beta_coeffs"))
)
previous_params = Float64.(
    vec(h5read(PREVIOUS_RESULTS_H5, "reconstruction/params_vector"))
)

previous_target_A_raw = Float64.(
    vec(h5read(PREVIOUS_RESULTS_H5, "targets/target_A_raw"))
)
previous_target_phi_raw = Float64.(
    vec(h5read(PREVIOUS_RESULTS_H5, "targets/target_phi_raw"))
)
previous_target_A_used = Float64.(
    vec(h5read(PREVIOUS_RESULTS_H5, "targets/target_A_used"))
)
previous_target_phi_used = Float64.(
    vec(h5read(PREVIOUS_RESULTS_H5, "targets/target_phi_used"))
)

previous_timestamps_unix = Float64.(
    vec(h5read(PREVIOUS_RESULTS_H5, "time/timestamps_unix_used"))
)
previous_indices_used = Int.(
    vec(h5read(PREVIOUS_RESULTS_H5, "time/indices_used"))
)
previous_ground_indices = Int.(
    vec(h5read(PREVIOUS_RESULTS_H5, "ground/indices_TX_to_RX"))
)

length(previous_h_coeffs) == 9 || error(
    "La BBO-1 previa debe contener 9 coeficientes de h'."
)
length(previous_beta_coeffs) == 6 || error(
    "La BBO-1 previa debe contener 6 coeficientes de beta."
)
length(previous_params) == 15 || error(
    "La BBO-1 previa debe contener 15 parámetros."
)
all(isfinite, previous_params) || error(
    "La semilla BBO-1 contiene NaN o Inf."
)

previous_ground_indices == GROUND_INDICES_TX_TO_RX || error(
    "El suelo del HDF5 BBO-1 no coincide con la configuración 5+5 de este script."
)

maximum(
    abs.(
        previous_params .-
        vcat(previous_h_coeffs, previous_beta_coeffs)
    )
) <= 1e-12 || error(
    "reconstruction/params_vector no coincide con los coeficientes h'/beta del HDF5 BBO-1."
)

coeffs_h_alb = copy(previous_h_coeffs)
coeffs_β_alb = copy(previous_beta_coeffs)

p_β_alb = Polynomial(coeffs_β_alb)
p_h_alb = Polynomial(coeffs_h_alb)
pars = vcat(p_h_alb, p_β_alb)

pars_list = map(coeffs, pars)
nh = length(coeffs_h_alb)

println("Semilla del restart cargada desde:")
println("  ", PREVIOUS_RESULTS_H5)
println("  h' coefficients: ", coeffs_h_alb)
println("  beta coefficients: ", coeffs_β_alb)

using Optim, Statistics

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
# Auditoría estricta del restart
#################################################################################################
length(previous_target_A_raw) == length(target_A_raw) || error(
    "Longitud incompatible en target_A_raw entre la BBO-1 y esta BBO-2."
)
length(previous_target_phi_raw) == length(target_phi_raw) || error(
    "Longitud incompatible en target_phi_raw entre la BBO-1 y esta BBO-2."
)
length(previous_timestamps_unix) == length(timestamps_test) || error(
    "Longitud incompatible en timestamps entre la BBO-1 y esta BBO-2."
)
length(previous_indices_used) == length(indices) || error(
    "Longitud incompatible en indices entre la BBO-1 y esta BBO-2."
)

maximum(abs.(previous_target_A_raw .- target_A_raw)) <= 1e-12 || error(
    "El target crudo de amplitud no coincide exactamente con la BBO-1 válida."
)
maximum(abs.(previous_target_phi_raw .- target_phi_raw)) <= 1e-12 || error(
    "El target crudo de fase no coincide exactamente con la BBO-1 válida."
)
maximum(abs.(previous_target_A_used .- target_A_used)) <= 1e-12 || error(
    "El target corregido de amplitud no coincide con la BBO-1 válida."
)
maximum(abs.(previous_target_phi_used .- target_phi_used)) <= 1e-12 || error(
    "El target corregido de fase no coincide con la BBO-1 válida."
)
maximum(abs.(previous_timestamps_unix .- timestamps_test)) <= 1e-6 || error(
    "Los tiempos de inversión no coinciden con la BBO-1 válida."
)
previous_indices_used == indices || error(
    "Los índices de inversión no coinciden con la BBO-1 válida."
)

maximum(
    abs.(
        (previous_target_A_used .- previous_target_A_raw) .-
        AMP_OFFSET_DB_USED
    )
) <= 1e-12 || error(
    "El HDF5 BBO-1 no usa el offset de amplitud +19 dB."
)

maximum(
    abs.(
        (previous_target_phi_used .- previous_target_phi_raw) .-
        PHI_OFFSET_DEG_USED
    )
) <= 1e-12 || error(
    "El HDF5 BBO-1 no usa el offset de fase -210 deg."
)

println("Auditoría del restart superada:")
println("  target: +19 dB, -210 deg")
println("  tiempos e índices: idénticos a la BBO-1 válida")
println("  suelo: ocean5_goodsoil5")

#################################################################################################
# Cache LMP
#################################################################################################
const CACHE = build_cache(
    ts_data,
    tx2,
    rx,
    rx_rec,
    GROUND_BY_SEGMENT,
    dists_new,
)

length(CACHE.grounds) == 10 || error(
    "The inversion cache must contain exactly 10 ground entries."
)


using LinearAlgebra
BLAS.set_num_threads(1)                       # evita oversubscribe con OpenBLAS/MKL
const NTH = max(1, Threads.nthreads()-1)

#################################################################################################
# Escalas de normalización
#################################################################################################
scale_A = sum((target_A_used .- mean(target_A_used)).^2) + 1e-12
scale_phi = sum((target_phi_used .- mean(target_phi_used)).^2) + 1e-12

function normalized_loss_from_models(A_model, phi_model, target_A_used, target_phi_used, scale_A, scale_phi)
    error_A = sum((A_model .- target_A_used).^2)
    error_phi = sum((phi_model .- target_phi_used).^2)
    χ2 = error_A / scale_A + error_phi / scale_phi
    return isfinite(χ2) ? χ2 : 1e99
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

    println("Norm. error_A: ", error_A/scale_A, ", norm. error_ϕ: ", error_phi/scale_phi)

    χ2 = error_A / scale_A + error_phi / scale_phi
    return isfinite(χ2) ? χ2 : 1e99
end

loss_wrapper(pp) = loss_bg(pp, mat_sza, target_A_used, target_phi_used, CACHE)

#################################################################################################
# Configuración de la inversión
#################################################################################################
const MAXTIME_SEC = 10.0 * 60 * 60

const INITIAL_SCALE = 1.0
const SEARCH_LOW_FACTOR = 0.85
const SEARCH_HIGH_FACTOR = 1.15

pars0 = vcat(pars_list[1].*1.0, pars_list[2].*1.0)
initial_params = INITIAL_SCALE .* pars0

lower = SEARCH_LOW_FACTOR .* initial_params
upper = SEARCH_HIGH_FACTOR .* initial_params

using BlackBoxOptim
using Logging
# global_logger(NullLogger())

search_range = [(min(lower[i], upper[i]), max(lower[i], upper[i])) for i in 1:length(upper)]
num_params = length(initial_params)
nthreads = max(1, Threads.nthreads()-1)  # evita 0

#################################################################################################
# Evaluar la reconstrucción BBO-1 previa antes del restart BBO-2
#################################################################################################
pp_seed = [initial_params[1:nh], initial_params[nh+1:end]]
A_seed_model, phi_seed_model = vlfsignal_simple_perturbation_v2(
    zdt_ref,
    phi_firi_wrapped_ref,
    mat_sza,
    pp_seed,
    CACHE,
)

seed_loss = normalized_loss_from_models(
    A_seed_model,
    phi_seed_model,
    target_A_used,
    target_phi_used,
    scale_A,
    scale_phi,
)

println("chi2(seed hetero BBO-1) = ", seed_loss)

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

reco_loss = normalized_loss_from_models(
    A_reco_model,
    phi_reco_model,
    target_A_used,
    target_phi_used,
    scale_A,
    scale_phi,
)

χ2_init = seed_loss
χ2_best = reco_loss

println("chi2(init) = ", χ2_init)
println("chi2(best) = ", χ2_best)
println("best_fitness(BBO) = ", best_fitness(result))

#################################################################################################
# Guardado limpio: semilla BBO-1 válida + reconstrucción BBO-2
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

    reco_h_coeffs,
    reco_beta_coeffs,
    reco_params,
    A_reco_model,
    phi_reco_model,
    reco_loss,

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
    loss_type = "normalized_chi2_A_phi",
    target_source = "A_background, phi_background",
)
    h5open(output_h5, "w") do f
        # ----------------------------------------------------
        # Global metadata
        # ----------------------------------------------------
        HDF5.attributes(f)["method"] = "real_background_inversion_from_separation_h5"
        HDF5.attributes(f)["source_separation_h5"] = String(source_separation_h5)
        HDF5.attributes(f)["target_source"] = target_source
        HDF5.attributes(f)["loss_type"] = loss_type
        HDF5.attributes(f)["bbo_method"] = String(bbo_method)
        HDF5.attributes(f)["amp_offset_db_used"] = amp_offset_db_used
        HDF5.attributes(f)["phi_offset_deg_used"] = phi_offset_deg_used
        HDF5.attributes(f)["initial_scale"] = initial_scale
        HDF5.attributes(f)["search_low_factor"] = search_low_factor
        HDF5.attributes(f)["search_high_factor"] = search_high_factor
        HDF5.attributes(f)["created_at"] = string(now())
        HDF5.attributes(f)["parameter_names"] = join(parameter_names, ",")
        HDF5.attributes(f)["ground_case_key"] = GROUND_CASE_KEY
        HDF5.attributes(f)["ground_order"] = "TX_to_RX"
        HDF5.attributes(f)["phase_reference_jld2"] = String(filename_ref)
        HDF5.attributes(f)["phase_unwrap_method"] = (
            "DSP.unwrap on the complete candidate series, followed by one " *
            "global integer-2pi branch shift relative to the independently " *
            "unwrapped FIRI reference for the same ground configuration."
        )
        HDF5.attributes(f)["phase_alignment_to_homogeneous_seawater"] = false
        HDF5.attributes(f)["restart_stage"] = "BBO-2"
        HDF5.attributes(f)["previous_results_h5"] = String(PREVIOUS_RESULTS_H5)
        HDF5.attributes(f)["initial_source"] = "valid_heterogeneous_BBO1_reconstruction_16592871"
        HDF5.attributes(f)["bounds_source"] = "plus_minus_15_percent_around_previous_heterogeneous_BBO1_reconstruction"

        # ----------------------------------------------------
        # Ground configuration
        # ----------------------------------------------------
        gg = create_group(f, "ground")
        write(
            gg,
            "indices_TX_to_RX",
            Int.(GROUND_INDICES_TX_TO_RX),
        )
        write(
            gg,
            "epsilon_r_by_segment",
            Float64.(GROUND_EPSILON_R_BY_SEGMENT),
        )
        write(
            gg,
            "sigma_S_m_by_segment",
            Float64.(GROUND_SIGMA_S_M_BY_SEGMENT),
        )
        HDF5.attributes(gg)["description"] = (
            "Ten lower-boundary materials ordered from NAA (TX) to PLO (RX): " *
            "five GROUND[10] sea-water segments followed by five GROUND[7] " *
            "good-soil segments."
        )

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
        # Previous heterogeneous BBO-1 seed
        # ----------------------------------------------------
        gs = create_group(f, "seed")
        write(gs, "h_coeffs", Float64.(seed_h_coeffs))
        write(gs, "beta_coeffs", Float64.(seed_beta_coeffs))
        write(gs, "params_vector", Float64.(seed_params))
        write(gs, "A_model", Float64.(A_seed_model))
        write(gs, "phi_model", Float64.(phi_seed_model))
        HDF5.attributes(gs)["loss"] = Float64(seed_loss)
        HDF5.attributes(gs)["description"] = "Forward model evaluated at the previous valid heterogeneous BBO-1 reconstruction used as the BBO-2 seed."

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

const INVERSION_OUTPUT_DIR = joinpath(
    ROOT_DIR,
    "outputs_real_background_inversion_whittaker_ocean5_goodsoil5_BBO2_restart15",
)
mkpath(INVERSION_OUTPUT_DIR)

jobid = get(ENV, "SLURM_JOB_ID", "nojid")
filename_hdf5 = joinpath(
    INVERSION_OUTPUT_DIR,
    "new-real-background-inversion_whittaker_ocean5_goodsoil5_BBO2_restart15_results_$(jobid).h5",
)

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

    seed_h_coeffs = pars_list[1],
    seed_beta_coeffs = pars_list[2],
    seed_params = initial_params,
    A_seed_model = A_seed_model,
    phi_seed_model = phi_seed_model,
    seed_loss = seed_loss,

    reco_h_coeffs = best_params[1:nh],
    reco_beta_coeffs = best_params[nh+1:end],
    reco_params = reco_params,
    A_reco_model = A_reco_model,
    phi_reco_model = phi_reco_model,
    reco_loss = reco_loss,

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
    target_source = "A_background, phi_background",
)
