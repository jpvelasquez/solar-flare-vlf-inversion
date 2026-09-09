#####################################################################################
#################### Loading Julia files needed ##  ###################################
#####################################################################################
################# preamble.jl contains all the packages necessary in this simulation.
include("./mytools/preamble.jl") 
##########################################################################################################################################
######### scenario.jl contains information on TX and RX position,
######### their features (VerticalDipole and GroundSampler respectively),
######### geomagnetic field, altitude, initial and final time, etc. ######################################################################################
include("./mytools/scenario.jl")
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
    GMT.coast(region=:global, xaxis=(annot=60,), yaxis=(annot=30,), proj=(name=:ortho, center=mid_point2), 
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

        wg2 = SegmentedWaveguide([HomogeneousWaveguide(bf_k[j], species2[j],
                  cache.ground, cache.dists12[j]) for j in 1:Nwaveguide])

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
using HDF5
function save_results_h5(filename_hdf5;
    timestamps, target_A, target_phi, A_reco, phi_reco,
    pars_list::Vector{<:AbstractVector},
    initial_params,
    reco_params,
    reco_model::AbstractString = "unknown",
    parameter_names::Vector{String} = String[],
    background_source_h5::AbstractString = "unknown",
    background_source_type::AbstractString = "unknown",
    amp_offset_db_used::Real = NaN,
    phi_offset_deg_used::Real = NaN)

    ts   = collect(Float64.(timestamps))
    tA   = collect(Float64.(target_A))
    tPhi = collect(Float64.(target_phi))
    rA   = collect(Float64.(A_reco))
    rPhi = collect(Float64.(phi_reco))

    @assert length(pars_list) >= 2 "pars_list debe tener al menos 2 vectores"
    true_h    = collect(Float64.(pars_list[1]))
    true_beta = collect(Float64.(pars_list[2]))

    init_vec = initial_params isa Real ? [Float64(initial_params)] : collect(Float64.(initial_params))
    reco_vec = reco_params isa Real ? [Float64(reco_params)] : collect(Float64.(reco_params))
    pnames   = isempty(parameter_names) ? ["p$(i)" for i in eachindex(reco_vec)] : parameter_names
    @assert length(pnames) == length(reco_vec) "parameter_names y reco_params deben tener la misma longitud"
    @assert length(init_vec) == length(reco_vec) "initial_params y reco_params deben tener la misma longitud"

    h5open(filename_hdf5, "w") do f
        write(f, "timestamps", ts)
        write(f, "target_A",   tA)
        write(f, "target_phi", tPhi)
        write(f, "A_reco",     rA)
        write(f, "phi_reco",   rPhi)
        write(f, "true_h",        true_h)
        write(f, "true_beta",     true_beta)
        write(f, "initial_params", init_vec)
        write(f, "reco_params",   reco_vec)

        HDF5.attributes(f)["reco_model"] = reco_model
        HDF5.attributes(f)["parameter_names"] = join(pnames, ",")
        HDF5.attributes(f)["background_source_h5"] = background_source_h5
        HDF5.attributes(f)["background_source_type"] = background_source_type
        HDF5.attributes(f)["amp_offset_db_used"] = Float64(amp_offset_db_used)
        HDF5.attributes(f)["phi_offset_deg_used"] = Float64(phi_offset_deg_used)
    end
end

############################################################################################
# === LECTOR PARA archivos creados con save_results_h5 ===
using HDF5

struct InversionResults
    timestamps :: Vector{Float64}
    target_A   :: Vector{Float64}
    target_phi :: Vector{Float64}
    A_reco     :: Vector{Float64}
    phi_reco   :: Vector{Float64}
    true_h     :: Vector{Float64}
    true_beta  :: Vector{Float64}
    initial_params :: Vector{Float64}
    reco_params :: Vector{Float64}
    reco_model :: String
    parameter_names :: Vector{String}
    meta       :: Dict{String,String}
end

function load_results_h5(path::AbstractString)::InversionResults
    @assert isfile(path) "No existe el archivo: $path"
    h5open(path, "r") do f
        ts   = Vector{Float64}(read(f["timestamps"]))
        tA   = Vector{Float64}(read(f["target_A"]))
        tPhi = Vector{Float64}(read(f["target_phi"]))
        rA   = Vector{Float64}(read(f["A_reco"]))
        rPhi = Vector{Float64}(read(f["phi_reco"]))
        th   = Vector{Float64}(read(f["true_h"]))
        tb   = Vector{Float64}(read(f["true_beta"]))
        ip   = haskey(f, "initial_params") ? Vector{Float64}(read(f["initial_params"])) : Float64[]
        rp   = haskey(f, "reco_params") ? Vector{Float64}(read(f["reco_params"])) : Float64[]

        meta = Dict{String,String}()
        reco_model = "unknown"
        parameter_names = String[]
        ats = HDF5.attributes(f)
        for k in keys(ats)
            val = try
                read(ats[k])
            catch
                nothing
            end
            meta[String(k)] = string(val)
        end
        reco_model = get(meta, "reco_model", "unknown")
        parameter_names = isempty(get(meta, "parameter_names", "")) ? String[] : split(meta["parameter_names"], ",")

        return InversionResults(ts, tA, tPhi, rA, rPhi, th, tb, ip, rp, reco_model, parameter_names, meta)
    end
end

##############################################################################################
function summary(r::InversionResults)
    println("InversionResults")
    println("  N timestamps:      ", length(r.timestamps))
    println("  target_A / phi:    ", length(r.target_A), " / ", length(r.target_phi))
    println("  reco_A / phi:      ", length(r.A_reco), " / ", length(r.phi_reco))
    println("  true_h / true_beta:", length(r.true_h), " / ", length(r.true_beta))
    println("  reco_model:        ", r.reco_model)
    println("  initial_params:    ", r.initial_params)
    println("  reco_params:       ", r.reco_params)
    println("  parameter_names:   ", r.parameter_names)
    if !isempty(r.meta)
        println("  meta: ", r.meta)
    end
    nothing
end
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
using HDF5
using Dates
using TimeZones
filename_asls = "datos_asls_full-v3.h5"
# 1) Lee directamente cada arreglo con h5read
baseline_asls_amp    = h5read(filename_asls, "baseline_asls_amp")
baseline_asls_phi    = h5read(filename_asls, "baseline_asls_phi")
detrended_asls_amp   = h5read(filename_asls, "detrended_asls_amp")
detrended_asls_phi   = h5read(filename_asls, "detrended_asls_phi")
time_unix            = h5read(filename_asls, "time_unix")

# 2) Convierte UNIX → ZonedDateTime
#   Asumimos time_unix en segundos desde epoch UTC.
epoch      = DateTime(1970,1,1)
dt_utc     = epoch .+ Second.(time_unix)       # Vector{DateTime} en UTC
tz_utc    = tz"UTC"                  # tu zona local
time_zoned = ZonedDateTime.(dt_utc, tz_utc)   # Vector{ZonedDateTime}
#################################################################################################################################
#### Renombrando los arreglos:
#################################################################################################################################
baseline_amp = baseline_asls_amp
baseline_phi = baseline_asls_phi
amp_detrended = detrended_asls_amp
phi_detrended = detrended_asls_phi
time_dt = Date.(time_zoned)
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
###################################################################################################################################
############################  Funciones y parámetros de la perturbación    ########################################################
###################################################################################################################################
t₀ = ZonedDateTime(2008, 3, 25, 18, 30, 0, tz"UTC")

@inline nexp(z_km::Real, z0::Real, hz::Real) = exp(clamp((z_km - z0) / hz, -50.0, 50.0))

@inline function gauss_asym_F3(Δt::Real, amp::Real, t_pk::Real, τ_u::Real, τ_d::Real, eps_t::Real)
    w = τ_u + 0.5 * (τ_d - τ_u) * (1.0 + tanh((Δt - t_pk) / eps_t))
    return amp * exp(-0.5 * ((Δt - t_pk) / w)^2)
end

@inline function ne_perturb_f3(z::Real, h_i::Real, β_i::Real, Δt::Real,
                               amp::Real, t_pk::Real, τ_u::Real, τ_d::Real, eps_t::Real,
                               z0::Real, hz::Real)
    ne_bg = waitprofile(z, h_i, β_i)
    return ne_bg * (1 + nexp(z/1e3, z0, hz) * gauss_asym_F3(Δt, amp, t_pk, τ_u, τ_d, eps_t))
end

function vlfsignal_perturbation_f3(zdt_test_ref, ϕ_firi_wrapped_ref, mat_sza, pars_list,
                                   p::AbstractVector{<:Real}, cache::SimCache)
    @assert length(p) == 7 "F3+nexp requiere 7 parámetros: [amp, t_pk, τ_u, τ_d, eps_t, z0, hz]"
    amp, t_pk, τ_u, τ_d, eps_t, z0, hz = p

    p_h_sza = Polynomial(pars_list[1])
    p_β_sza = Polynomial(pars_list[2])

    Nt = length(cache.dt)
    A_fit_sfₜ = Vector{Float64}(undef, Nt)
    ϕ_fit_sfₜ = Vector{Float64}(undef, Nt)

    @inbounds for (k, tt) in enumerate(cache.dt)
        Δt = Second(tt - t₀).value
        sza_tᵢ     = mat_sza[k, :]
        rad_sza_tᵢ = (π/180.0) .* sza_tᵢ
        hs_aux     = p_h_sza.(rad_sza_tᵢ[2:2:end])
        βs_aux     = p_β_sza.(rad_sza_tᵢ[2:2:end])
        Nwaveguide = length(sza_tᵢ[2:2:end])

        bf_k = cache.bf_list[k]
        species_sf = Vector{Any}(undef, Nwaveguide)
        for i in 1:Nwaveguide
            h_i, β_i = hs_aux[i], βs_aux[i]
            f_perturb = z -> ne_perturb_f3(z, h_i, β_i, Δt, amp, t_pk, τ_u, τ_d, eps_t, z0, hz)
            species_sf[i] = Species(QE, ME, f_perturb, electroncollisionfrequency)
        end

        wg_sf = SegmentedWaveguide([
            HomogeneousWaveguide(bf_k[j], species_sf[j], cache.ground, cache.dists12[j])
            for j in 1:Nwaveguide
        ])
        _, A_fit_seg_sf, ϕ_fit_seg_sf = propagate(wg_sf, cache.tx, cache.rx)
        A_fit_sfₜ[k] = Float64(A_fit_seg_sf)
        ϕ_fit_sfₜ[k] = Float64(ϕ_fit_seg_sf)
    end

    ϕ_fit_sf_unₜ = unwrap_resampled(zdt_test_ref, ϕ_firi_wrapped_ref, cache.dt, ϕ_fit_sfₜ)
    ϕ_fit_sf_unₜ = rad2deg.(ϕ_fit_sf_unₜ)
    return A_fit_sfₜ, ϕ_fit_sf_unₜ
end
####################################################################################################################################
### Generando trayectorias y objeto "Receiver" para usar IGRF ######################################################################
###################################################################################################################################
npoints = 19#19#5#6#3#12#20
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

mat_sza = Matrix{Float64}(undef,size(zdt_test)[1],size(dists_new)[1])

for (k,tt) in enumerate(zdt_test)
    #sza_temp = get_sza.(lats_new, lons_new, tt)
    sza_temp = chi_signed_astrolib.(lats_new, lons_new, tt)#
    mat_sza[k,:] = sza_temp
end

### Background h' and beta polynomials are loaded from the selected old-ASLS real-background inversion.
### No h/beta coefficients are hardcoded here.
# Background coefficients loaded from the selected real-background inversion.
# Selected run: old ASLS restart / second 10 h batch with best normalized chi2.
# This replaces the previous hardcoded h' and beta coefficients.
const BACKGROUND_H5 = get(
    ENV,
    "BACKGROUND_H5",
    "new-real-background-inversion-bbo_from_old_asls_restart_results_13341497.h5",
)

function load_background_coeffs_from_h5(path::AbstractString)
    @assert isfile(path) "No existe BACKGROUND_H5: $(path). Pon el .h5 en el directorio de trabajo o define ENV[\"BACKGROUND_H5\"]."

    h_coeffs = Vector{Float64}(h5read(path, "reconstruction/h_coeffs"))
    β_coeffs = Vector{Float64}(h5read(path, "reconstruction/beta_coeffs"))

    @assert length(h_coeffs) == 9 "Se esperaban 9 coeficientes de h'; encontré $(length(h_coeffs))."
    @assert length(β_coeffs) == 6 "Se esperaban 6 coeficientes de beta; encontré $(length(β_coeffs))."

    # Verificación opcional contra optimization/best_candidate si existe.
    try
        best_candidate_bg = Vector{Float64}(h5read(path, "optimization/best_candidate"))
        @assert length(best_candidate_bg) == 15 "optimization/best_candidate debe tener 15 elementos."
        maxdiff_h = maximum(abs.(best_candidate_bg[1:9] .- h_coeffs))
        maxdiff_β = maximum(abs.(best_candidate_bg[10:15] .- β_coeffs))
        @assert maxdiff_h < 1e-10 "h_coeffs no coincide con best_candidate[1:9]. maxdiff=$(maxdiff_h)"
        @assert maxdiff_β < 1e-10 "beta_coeffs no coincide con best_candidate[10:15]. maxdiff=$(maxdiff_β)"
        println("Verificación OK: reconstruction/*_coeffs coincide con optimization/best_candidate.")
    catch err
        println("Advertencia: no pude verificar contra optimization/best_candidate: ", err)
    end

    return h_coeffs, β_coeffs
end

coeffs_h_alb, coeffs_β_alb = load_background_coeffs_from_h5(BACKGROUND_H5)

println("Background h/beta cargado desde: ", BACKGROUND_H5)
println("coeffs_h_alb = ", coeffs_h_alb)
println("coeffs_β_alb = ", coeffs_β_alb)

p_β_alb = Polynomial(coeffs_β_alb)
p_h_alb = Polynomial(coeffs_h_alb)
pars = vcat(p_h_alb,p_β_alb)

pars_list = map(coeffs, pars)
nh = length(coeffs_h_alb)

using Optim, Statistics

ts_data = timestamps_test
# correcciones instrumentales/datos
target_A   = baseline_amp[indices] .+ detrended_asls_amp[indices]
target_phi = baseline_phi[indices] .+ detrended_asls_phi[indices]

##################################################
const CACHE = build_cache(ts_data, tx2, rx, rx_rec, ground, dists_new)
using LinearAlgebra
BLAS.set_num_threads(1)
const NTH = max(1, Threads.nthreads()-1)
################################################################################################
# Estimating normalizing scales:
#################################################################################################
# fuera del loss, una sola vez

target_A_corr = target_A .+ 19.0
target_phi_corr = target_phi .- 210.0

scale_A   = sum((target_A_corr   .- mean(target_A_corr)).^2) + 1e-12
scale_phi = sum((target_phi_corr .- mean(target_phi_corr)).^2) + 1e-12
#################################################################################################
# LOSS para F3
#################################################################################################
@inline function loss_f3(p, pars_list, mat_sza, target_A, target_phi, cache::SimCache)
    @assert length(p) == 7 "F3+nexp requiere 7 parámetros"
    amp, t_pk, τ_u, τ_d, eps_t, z0, hz = p

    if !all(isfinite, p) || amp < 0 || τ_u <= 0 || τ_d <= 0 || eps_t <= 0 || hz <= 0
        return 1e99
    end

    target_A_corr   = target_A .+ 19.0
    target_phi_corr = target_phi .- 210.0

    Aₜ, ϕₜ = try
        vlfsignal_perturbation_f3(zdt_ref, phi_firi_wrapped_ref, mat_sza, pars_list, p, cache)
    catch
        return 1e99
    end

    if any(!isfinite, Aₜ) || any(!isfinite, ϕₜ)
        return 1e99
    end

    @inbounds error_A = sum((Aₜ[i] - target_A_corr[i])^2 for i in eachindex(target_A_corr))
    @inbounds rϕ = target_phi_corr .- ϕₜ
    @inbounds error_phi = sum((rϕ[i])^2 for i in eachindex(rϕ))
    println("Norm. error_A: ",error_A/scale_A, ", nor. error_ϕ: ", error_phi/scale_phi)#, ", ratio: ", error_ϕ/error_A)
    χ2 = error_A / scale_A + error_phi / scale_phi
#    χ2 = error_A + error_ϕi
    
    return isfinite(χ2) ? χ2 : 1e99
end

loss_wrapper(p) = loss_f3(p, pars_list, mat_sza, target_A, target_phi, CACHE)

#####################################################################################################
# Semilla F3 desde el ajuste Python (mejor bg grado 4)
# Python: tiempos en horas desde 25-Mar-2008 17:00 UTC
#####################################################################################################
const MAXTIME_SEC = 10.0 * 60 * 60#0.5 * 60 * 60

tref_py = ZonedDateTime(2008, 3, 25, 17, 0, 0, tz"UTC")
offset_s = Dates.value(DateTime(t₀) - DateTime(tref_py)) / 1000.0

amp0   = 20.0#3.7592316
t_pk0  = 1.9470757   * 3600.0 - offset_s
τ_u0   = 0.052938374 * 3600.0
τ_d0   = 0.34682182  * 3600.0
eps0   = 0.21935698  * 3600.0
z0_0   = 85.0
hz_0   = 10.0

pars0 = [amp0, t_pk0, τ_u0, τ_d0, eps0, z0_0, hz_0]
#lower = [1.0, 600.0,  60.0, 300.0, 120.0, 70.0,  8.0]
#upper = [8.0, 3600.0, 1200.0, 5000.0, 2500.0, 90.0, 12.0]

lower = [15, 600.0,  60.0, 300.0, 120.0, 70.0,  8.0]
upper = [25, 3600.0, 1200.0, 5000.0, 2500.0, 90.0, 12.0]


initial_params = pars0
using BlackBoxOptim
using Logging
search_range = [(lower[i], upper[i]) for i in eachindex(lower)]
num_params = length(initial_params)
nthreads = max(1, Threads.nthreads()-1)

result = BlackBoxOptim.bboptimize(loss_wrapper, initial_params;
                 Method        = :dxnes,
                 SearchRange   = search_range,
                 NumDimensions = num_params,
                 MaxFuncEvals  = 1e9,
                 TraceMode     = :compact,
                 MaxTime       = MAXTIME_SEC,
                 NThreads      = nthreads)

println("Mejor solución encontrada: ", best_candidate(result))
println("Mejor error encontrado: ", best_fitness(result))
println("Parámetros iniciales: ", initial_params)

best_params = collect(Float64.(best_candidate(result)))
reco_params = best_params
println("Parámetros óptimos F3+nexp [amp, t_pk, τ_u, τ_d, eps, z0, hz]: ", reco_params)

A_reco, ϕ_reco = vlfsignal_perturbation_f3(zdt_ref, phi_firi_wrapped_ref, mat_sza, pars_list, reco_params, CACHE)

jobid = get(ENV, "SLURM_JOB_ID", "nojid")
filename_hdf5 = "new-perturbation-inversion_F3_nexp_oldASLS_BG13341497_results_$(jobid).h5"
save_results_h5(filename_hdf5;
    timestamps       = timestamps_test,
    target_A         = target_A,
    target_phi       = target_phi,
    A_reco           = A_reco,
    phi_reco         = ϕ_reco,
    pars_list        = pars_list,
    reco_params      = reco_params,
    initial_params   = initial_params,
    reco_model       = "F3_gaussiana_asimetrica_nexp_vertical",
    parameter_names  = ["amp", "t_pk_s", "tau_u_s", "tau_d_s", "eps_s", "z0_km", "hz_km"],
    background_source_h5 = BACKGROUND_H5,
    background_source_type = "old_ASLS_restart_13341497_reconstruction_h_beta_coeffs",
    amp_offset_db_used = 19.0,
    phi_offset_deg_used = -210.0
)
