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
    separation_source_h5::AbstractString = "unknown",
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
        HDF5.attributes(f)["separation_source_h5"] = separation_source_h5
        HDF5.attributes(f)["background_source_h5"] = background_source_h5
        HDF5.attributes(f)["background_source_type"] = background_source_type
        HDF5.attributes(f)["background_family"] = "Whittaker/arPLS separation 16-22 restart15"
        HDF5.attributes(f)["old_asls_used"] = "false"
        HDF5.attributes(f)["target_source"] = "A_real, phi_real from separation H5"
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

# --------------------------------------------------------------------------------------------------
# Separación Whittaker/arPLS seleccionada para el evento real 2008-03-25, 16:00--22:00 UTC.
#
# IMPORTANTE:
#   - Este script NO usa datos_asls_full-v3.h5.
#   - Los targets reales vienen de A_real y phi_real de este HDF5.
#   - En esta versión restart, t_pk fijo se hereda de PREVIOUS_RESULTS_H5.
#   - La primera corrida también guardó el score calculado desde A_perturbation_raw y phi_perturbation_raw.
# --------------------------------------------------------------------------------------------------
const SEPARATION_H5 = get(
    ENV,
    "SEPARATION_H5",
    "real_whittaker_selected_background_16_22_for_inversion.h5",
)

function load_whittaker_separation_h5(path::AbstractString)
    @assert isfile(path) "No existe SEPARATION_H5: $(path). Pon el .h5 en el directorio de trabajo o define ENV[\"SEPARATION_H5\"]."

    timestamps_unix = Vector{Int64}(h5read(path, "timestamps_unix"))
    A_real          = Vector{Float64}(h5read(path, "A_real"))
    phi_real        = Vector{Float64}(h5read(path, "phi_real"))
    A_background    = Vector{Float64}(h5read(path, "A_background"))
    phi_background  = Vector{Float64}(h5read(path, "phi_background"))
    A_pert_raw      = Vector{Float64}(h5read(path, "A_perturbation_raw"))
    phi_pert_raw    = Vector{Float64}(h5read(path, "phi_perturbation_raw"))
    A_pert_fit      = Vector{Float64}(h5read(path, "A_perturbation"))
    phi_pert_fit    = Vector{Float64}(h5read(path, "phi_perturbation"))
    inversion_mask  = Bool.(h5read(path, "masks/inversion_mask_18_22"))

    n = length(timestamps_unix)
    @assert all(length(x) == n for x in (
        A_real, phi_real, A_background, phi_background,
        A_pert_raw, phi_pert_raw, A_pert_fit, phi_pert_fit, inversion_mask,
    )) "Las series del HDF5 de separación no tienen la misma longitud."

    maxdiff_A = maximum(abs.(A_real .- (A_background .+ A_pert_raw)))
    maxdiff_phi = maximum(abs.(phi_real .- (phi_background .+ phi_pert_raw)))
    @assert maxdiff_A < 1e-8 "No se cumple A_real ≈ A_background + A_perturbation_raw. maxdiff=$(maxdiff_A)"
    @assert maxdiff_phi < 1e-8 "No se cumple phi_real ≈ phi_background + phi_perturbation_raw. maxdiff=$(maxdiff_phi)"

    return (
        timestamps_unix = timestamps_unix,
        A_real = A_real,
        phi_real = phi_real,
        A_background = A_background,
        phi_background = phi_background,
        A_perturbation_raw = A_pert_raw,
        phi_perturbation_raw = phi_pert_raw,
        A_perturbation = A_pert_fit,
        phi_perturbation = phi_pert_fit,
        inversion_mask_18_22 = inversion_mask,
    )
end

sep = load_whittaker_separation_h5(SEPARATION_H5)

# Convierte UNIX → ZonedDateTime. Asumimos segundos desde epoch UTC.
time_unix  = sep.timestamps_unix
epoch      = DateTime(1970, 1, 1)
dt_utc     = epoch .+ Second.(time_unix)
tz_utc     = tz"UTC"
time_zoned = ZonedDateTime.(dt_utc, tz_utc)
time_dt    = Date.(time_zoned)

A_real = sep.A_real
phi_real = sep.phi_real
A_background_sep = sep.A_background
phi_background_sep = sep.phi_background
A_perturbation_raw_sep = sep.A_perturbation_raw
phi_perturbation_raw_sep = sep.phi_perturbation_raw
A_perturbation_fit_sep = sep.A_perturbation
phi_perturbation_fit_sep = sep.phi_perturbation
inversion_mask_18_22_sep = sep.inversion_mask_18_22

println("Separación Whittaker/arPLS cargada desde: ", SEPARATION_H5)
println("  n muestras separación = ", length(time_zoned))
println("  rango temporal         = ", first(time_zoned), "  →  ", last(time_zoned))
println("  puntos mask 18--22     = ", count(inversion_mask_18_22_sep))
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

### Background h' and beta polynomials are loaded from the selected Whittaker/arPLS-background inversion.
### No h/beta coefficients are hardcoded here.
# Selected run: restart15 / second batch from the 16--22 separation HDF5 target.
const BACKGROUND_H5 = get(
    ENV,
    "BACKGROUND_H5",
    "new-real-background-inversion_from_sep_16_22_restart15_results_12976567.h5",
)

# Primera corrida de perturbación Whittaker.
# Este archivo provee el seed inicial del restart y el mismo t_pk fijo.
const PREVIOUS_RESULTS_H5 = get(
    ENV,
    "PREVIOUS_RESULTS_H5",
    "new-perturbation-inversion_F3_nexp_whittakerBG12976567_FIXED_TPK_rel10_results_15517278.h5",
)

function load_previous_perturbation_restart_h5(path::AbstractString)
    @assert isfile(path) "No existe PREVIOUS_RESULTS_H5: $(path). Pon el .h5 previo en el directorio de trabajo o define ENV[\"PREVIOUS_RESULTS_H5\"]."

    free_params = Vector{Float64}(h5read(path, "reconstruction/free_params"))
    full_params = Vector{Float64}(h5read(path, "reconstruction/full_params"))
    t_pk_s      = Float64(h5read(path, "fixed/t_pk_s")[1])
    k_pk        = Int(h5read(path, "fixed/t_pk_index")[1])
    score_pk    = Vector{Float64}(h5read(path, "fixed/score_pk"))
    delta_A_pk  = Vector{Float64}(h5read(path, "fixed/delta_A_for_peak"))
    delta_phi_pk = Vector{Float64}(h5read(path, "fixed/delta_phi_for_peak"))
    sigma_A_pk  = Float64(h5read(path, "fixed/sigma_A_pk")[1])
    sigma_phi_pk = Float64(h5read(path, "fixed/sigma_phi_pk")[1])

    @assert length(free_params) == 6 "PREVIOUS_RESULTS_H5/reconstruction/free_params debe tener 6 parámetros libres."
    @assert length(full_params) == 7 "PREVIOUS_RESULTS_H5/reconstruction/full_params debe tener 7 parámetros completos."
    @assert isfinite(t_pk_s) "fixed/t_pk_s no es finito en PREVIOUS_RESULTS_H5."
    @assert 1 <= k_pk <= length(score_pk) "fixed/t_pk_index=$(k_pk) fuera de rango para score_pk."

    return (
        free_params = free_params,
        full_params = full_params,
        t_pk_s = t_pk_s,
        k_pk = k_pk,
        score_pk = score_pk,
        delta_A_pk = delta_A_pk,
        delta_phi_pk = delta_phi_pk,
        sigma_A_pk = sigma_A_pk,
        sigma_phi_pk = sigma_phi_pk,
    )
end

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

function read_root_attr_float(path::AbstractString, attr_name::AbstractString, default::Real)
    try
        h5open(path, "r") do f
            value = read(HDF5.attributes(f)[attr_name])
            if value isa AbstractArray
                return Float64(value[1])
            else
                return Float64(value)
            end
        end
    catch err
        println("Advertencia: no pude leer atributo $(attr_name) desde $(path). Usando default=$(default). Error: ", err)
        return Float64(default)
    end
end

function read_root_attr_string(path::AbstractString, attr_name::AbstractString, default::AbstractString)
    try
        h5open(path, "r") do f
            value = read(HDF5.attributes(f)[attr_name])
            return String(value)
        end
    catch
        return String(default)
    end
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


####################################################################################################
# INVERSION F3+nexp CON t_pk FIJO DESDE LOS DATOS
#
# Corrección respecto al script anterior:
#   - t_pk NO se optimiza.
#   - t_pk se hereda desde PREVIOUS_RESULTS_H5/fixed/t_pk_s.
#   - BBO optimiza solo 6 parámetros libres:
#       [amp, tau_u_s, tau_d_s, eps_s, z0_km, hz_km]
#   - El vector completo usado por vlfsignal_perturbation_f3 es:
#       [amp, t_pk_fixed_s, tau_u_s, tau_d_s, eps_s, z0_km, hz_km]
#   - El seed inicial se hereda desde PREVIOUS_RESULTS_H5/reconstruction/free_params.
#   - Las series del seed inicial se evalúan DESPUÉS de terminar BBO y se guardan en HDF5.
####################################################################################################

using Optim, Statistics
using LinearAlgebra
using HDF5
using Printf
using BlackBoxOptim
using Logging

BLAS.set_num_threads(1)

# --------------------------------------------------------------------------------------------------
# Targets usados por la inversión
# --------------------------------------------------------------------------------------------------

ts_data = timestamps_test

# Verificación temporal: los 17 puntos 18:00--22:00 deben caer dentro de la máscara de inversión 18--22.
@assert all(inversion_mask_18_22_sep[indices]) "Algún timestamp seleccionado no cae dentro de masks/inversion_mask_18_22 del HDF5 de separación."

# Datos reales usados en la inversión: señal completa real de la separación Whittaker/arPLS.
# No reconstruimos target como baseline+detrended; aquí el HDF5 ya trae A_real y phi_real.
target_A_raw   = A_real[indices]
target_phi_raw = phi_real[indices]

# Offsets instrumentales. Por defecto se leen del HDF5 de background Whittaker restart15.
const AMP_OFFSET_DB_USED = read_root_attr_float(BACKGROUND_H5, "amp_offset_db_used", 19.0)
const PHI_OFFSET_DEG_USED = read_root_attr_float(BACKGROUND_H5, "phi_offset_deg_used", -210.0)

target_A_corr   = target_A_raw   .+ AMP_OFFSET_DB_USED
target_phi_corr = target_phi_raw .+ PHI_OFFSET_DEG_USED

# Series auxiliares de separación evaluadas en los tiempos de inversión.
A_background_target_sep = A_background_sep[indices]
phi_background_target_sep = phi_background_sep[indices]
A_perturbation_raw_target_sep = A_perturbation_raw_sep[indices]
phi_perturbation_raw_target_sep = phi_perturbation_raw_sep[indices]
A_perturbation_fit_target_sep = A_perturbation_fit_sep[indices]
phi_perturbation_fit_target_sep = phi_perturbation_fit_sep[indices]

# --------------------------------------------------------------------------------------------------
# t_pk fijo y seed inicial heredados desde la primera corrida Whittaker.
# --------------------------------------------------------------------------------------------------

previous_restart = load_previous_perturbation_restart_h5(PREVIOUS_RESULTS_H5)

ΔA_pk = previous_restart.delta_A_pk
Δϕ_pk = previous_restart.delta_phi_pk
σA_pk = previous_restart.sigma_A_pk
σϕ_pk = previous_restart.sigma_phi_pk
score_pk = previous_restart.score_pk
k_pk = previous_restart.k_pk

@assert length(score_pk) == length(zdt_test) "score_pk previo tiene longitud $(length(score_pk)), pero zdt_test tiene $(length(zdt_test))."
@assert length(ΔA_pk) == length(zdt_test) "delta_A_for_peak previo tiene longitud incompatible."
@assert length(Δϕ_pk) == length(zdt_test) "delta_phi_for_peak previo tiene longitud incompatible."

zdt_pk_fixed = zdt_test[k_pk]
t_pk_fixed = previous_restart.t_pk_s
t_pk_from_index = Dates.value(DateTime(zdt_pk_fixed) - DateTime(t₀)) / 1000.0
@assert abs(t_pk_fixed - t_pk_from_index) < 1e-6 "t_pk_s previo=$(t_pk_fixed) no coincide con zdt_test[k_pk]-t₀=$(t_pk_from_index)."

println("\n================================================================================")
println("Restart: t_pk fijo heredado desde PREVIOUS_RESULTS_H5")
println("================================================================================")
println("  PREVIOUS_RESULTS_H5   = ", PREVIOUS_RESULTS_H5)
println("  k_pk                  = ", k_pk)
println("  instante pico usado   = ", zdt_pk_fixed)
println("  t₀                    = ", t₀)
println("  t_pk_fixed            = ", t_pk_fixed, " s respecto a t₀")
@printf("  score_pk[k_pk]        = %.8f\n", score_pk[k_pk])
@printf("  ΔA_pk[k_pk]           = %.8f dB\n", ΔA_pk[k_pk])
@printf("  Δϕ_pk[k_pk]           = %.8f deg\n", Δϕ_pk[k_pk])
@printf("  σA_pk                 = %.8f dB\n", σA_pk)
@printf("  σϕ_pk                 = %.8f deg\n", σϕ_pk)
println("================================================================================\n")

# --------------------------------------------------------------------------------------------------
# Cache y escalas de normalización del costo
# --------------------------------------------------------------------------------------------------

const CACHE = build_cache(ts_data, tx2, rx, rx_rec, ground, dists_new)
const NTH = max(1, Threads.nthreads() - 1)

scale_A   = sum((target_A_corr   .- mean(target_A_corr)).^2) + 1e-12
scale_phi = sum((target_phi_corr .- mean(target_phi_corr)).^2) + 1e-12

@printf("Escalas de normalización: scale_A=%.8e, scale_phi=%.8e\n", scale_A, scale_phi)

# --------------------------------------------------------------------------------------------------
# Parámetros: 6 libres + t_pk fijo
# --------------------------------------------------------------------------------------------------

const FREE_PARAMETER_NAMES = [
    "amp",
    "tau_u_s",
    "tau_d_s",
    "eps_s",
    "z0_km",
    "hz_km",
]

const FULL_PARAMETER_NAMES = [
    "amp",
    "t_pk_s_fixed",
    "tau_u_s",
    "tau_d_s",
    "eps_s",
    "z0_km",
    "hz_km",
]

function make_full_params_from_free(x_free::AbstractVector{<:Real}, tpk::Real)
    @assert length(x_free) == 6 "Se esperan 6 parámetros libres: [amp, tau_u, tau_d, eps, z0, hz]"
    amp, τ_u, τ_d, eps_t, z0, hz = Float64.(x_free)
    return [amp, Float64(tpk), τ_u, τ_d, eps_t, z0, hz]
end

function split_full_to_free(p_full::AbstractVector{<:Real})
    @assert length(p_full) == 7 "Se esperan 7 parámetros completos."
    return Float64[p_full[1], p_full[3], p_full[4], p_full[5], p_full[6], p_full[7]]
end

# Seed inicial del restart, leído desde la mejor reconstrucción de la primera corrida.
# Orden libre: [amp, tau_u_s, tau_d_s, eps_s, z0_km, hz_km]
initial_free_params = collect(Float64.(previous_restart.free_params))
initial_full_params = make_full_params_from_free(initial_free_params, t_pk_fixed)

maxdiff_prev_full = maximum(abs.(initial_full_params .- previous_restart.full_params))
@assert maxdiff_prev_full < 1e-8 "El full_params reconstruido desde free_params+t_pk no coincide con reconstruction/full_params previo. maxdiff=$(maxdiff_prev_full)"

println("Parámetros iniciales libres heredados [amp, tau_u, tau_d, eps, z0, hz]: ", initial_free_params)
println("Parámetros iniciales completos heredados [amp, t_pk_fixed, tau_u, tau_d, eps, z0, hz]: ", initial_full_params)

# Bounds de 6 dimensiones: restart dirigido.
#  - amp, tau_u, tau_d, eps, hz: ±10 %
#  - z0_km: ±20 %, porque en la primera corrida chocó contra el límite inferior.
const RELATIVE_SEARCH_FRACTION = 0.10
const RELATIVE_SEARCH_FRACTION_Z0 = 0.20
const RELATIVE_SEARCH_FRACTIONS = Float64[
    0.10,  # amp
    0.10,  # tau_u_s
    0.10,  # tau_d_s
    0.10,  # eps_s
    0.20,  # z0_km
    0.10,  # hz_km
]

lower_free = (1.0 .- RELATIVE_SEARCH_FRACTIONS) .* initial_free_params
upper_free = (1.0 .+ RELATIVE_SEARCH_FRACTIONS) .* initial_free_params

search_range = [(lower_free[i], upper_free[i]) for i in eachindex(lower_free)]
num_params = length(initial_free_params)

@assert num_params == 6
@assert length(search_range) == 6

println("SearchRange 6D:")
for i in eachindex(FREE_PARAMETER_NAMES)
    println("  ", FREE_PARAMETER_NAMES[i], ": ", search_range[i])
end

# --------------------------------------------------------------------------------------------------
# Loss 6D con t_pk fijo
# --------------------------------------------------------------------------------------------------

@inline function loss_f3_fixed_tpk(
    p_free,
    pars_list,
    mat_sza,
    target_A_corr,
    target_phi_corr,
    cache::SimCache,
    tpk::Real,
    scale_A::Real,
    scale_phi::Real,
)
    if length(p_free) != 6
        return 1e99
    end

    amp, τ_u, τ_d, eps_t, z0, hz = p_free

    if !all(isfinite, p_free) || amp < 0 || τ_u <= 0 || τ_d <= 0 || eps_t <= 0 || hz <= 0
        return 1e99
    end

    p_full = make_full_params_from_free(p_free, tpk)

    Aₜ, ϕₜ = try
        vlfsignal_perturbation_f3(zdt_ref, phi_firi_wrapped_ref, mat_sza, pars_list, p_full, cache)
    catch err
        return 1e99
    end

    if any(!isfinite, Aₜ) || any(!isfinite, ϕₜ)
        return 1e99
    end

    @inbounds error_A = sum((Aₜ[i] - target_A_corr[i])^2 for i in eachindex(target_A_corr))
    @inbounds error_phi = sum((ϕₜ[i] - target_phi_corr[i])^2 for i in eachindex(target_phi_corr))

    χ2_A = error_A / scale_A
    χ2_phi = error_phi / scale_phi
    χ2 = χ2_A + χ2_phi

    println("Norm. error_A: ", χ2_A, ", norm. error_ϕ: ", χ2_phi, ", total: ", χ2)

    return isfinite(χ2) ? χ2 : 1e99
end

loss_wrapper(p_free) = loss_f3_fixed_tpk(
    p_free,
    pars_list,
    mat_sza,
    target_A_corr,
    target_phi_corr,
    CACHE,
    t_pk_fixed,
    scale_A,
    scale_phi,
)

# --------------------------------------------------------------------------------------------------
# Optimización
# --------------------------------------------------------------------------------------------------

const MAXTIME_SEC = 10.0 * 60 * 60
nthreads = max(1, Threads.nthreads() - 1)

println("\n================================================================================")
println("Iniciando BBO 6D restart dirigido con t_pk fijo heredado")
println("================================================================================")
println("  MaxTime       = ", MAXTIME_SEC, " s")
println("  NumDimensions = ", num_params)
println("  NThreads      = ", nthreads)
println("  t_pk_fixed    = ", t_pk_fixed, " s")
println("  Previous H5   = ", PREVIOUS_RESULTS_H5)
println("================================================================================\n")

result = BlackBoxOptim.bboptimize(
    loss_wrapper,
    initial_free_params;
    Method        = :dxnes,
    SearchRange   = search_range,
    NumDimensions = num_params,
    MaxFuncEvals  = 1e9,
    TraceMode     = :compact,
    MaxTime       = MAXTIME_SEC,
    NThreads      = nthreads,
)

best_free_params = collect(Float64.(best_candidate(result)))
reco_free_params = best_free_params
reco_full_params = make_full_params_from_free(reco_free_params, t_pk_fixed)

println("Mejor solución libre encontrada [amp, tau_u, tau_d, eps, z0, hz]: ", reco_free_params)
println("Mejor solución completa [amp, t_pk_fixed, tau_u, tau_d, eps, z0, hz]: ", reco_full_params)
println("Mejor error encontrado: ", best_fitness(result))
println("Parámetros iniciales libres: ", initial_free_params)
println("Parámetros iniciales completos: ", initial_full_params)

# --------------------------------------------------------------------------------------------------
# Evaluación final de series seed y best DESPUÉS de BBO
# --------------------------------------------------------------------------------------------------

println("\nEvaluando serie temporal del seed inicial con t_pk fijo...")
A_seed, ϕ_seed = vlfsignal_perturbation_f3(
    zdt_ref,
    phi_firi_wrapped_ref,
    mat_sza,
    pars_list,
    initial_full_params,
    CACHE,
)

println("Evaluando serie temporal del best BBO con t_pk fijo...")
A_reco, ϕ_reco = vlfsignal_perturbation_f3(
    zdt_ref,
    phi_firi_wrapped_ref,
    mat_sza,
    pars_list,
    reco_full_params,
    CACHE,
)

A_seed = Float64.(A_seed)
ϕ_seed = Float64.(ϕ_seed)
A_reco = Float64.(A_reco)
ϕ_reco = Float64.(ϕ_reco)

# --------------------------------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------------------------------

function chi2_norm_local(model, target)
    den = sum((target .- mean(target)).^2) + 1e-12
    return sum((model .- target).^2) / den
end

function rmse_local(model, target)
    return sqrt(mean((model .- target).^2))
end

function metrics_local(A_model, phi_model, A_target, phi_target)
    chiA = chi2_norm_local(A_model, A_target)
    chiP = chi2_norm_local(phi_model, phi_target)
    return (
        chiA = chiA,
        chiP = chiP,
        chiT = chiA + chiP,
        rmseA = rmse_local(A_model, A_target),
        rmseP = rmse_local(phi_model, phi_target),
    )
end

m_seed = metrics_local(A_seed, ϕ_seed, target_A_corr, target_phi_corr)
m_reco = metrics_local(A_reco, ϕ_reco, target_A_corr, target_phi_corr)

println("\nMétricas seed inicial con t_pk fijo:")
@printf("  χ²_A      = %.8f\n", m_seed.chiA)
@printf("  χ²_ϕ      = %.8f\n", m_seed.chiP)
@printf("  χ²_total  = %.8f\n", m_seed.chiT)
@printf("  RMSE_A    = %.8f dB\n", m_seed.rmseA)
@printf("  RMSE_ϕ    = %.8f deg\n", m_seed.rmseP)

println("\nMétricas best BBO con t_pk fijo:")
@printf("  χ²_A      = %.8f\n", m_reco.chiA)
@printf("  χ²_ϕ      = %.8f\n", m_reco.chiP)
@printf("  χ²_total  = %.8f\n", m_reco.chiT)
@printf("  RMSE_A    = %.8f dB\n", m_reco.rmseA)
@printf("  RMSE_ϕ    = %.8f deg\n", m_reco.rmseP)

# --------------------------------------------------------------------------------------------------
# Guardado HDF5 auditado
# --------------------------------------------------------------------------------------------------

function write_metric_group!(parent, name::AbstractString, m)
    g = create_group(parent, name)
    write(g, "chi2_A", [Float64(m.chiA)])
    write(g, "chi2_phi", [Float64(m.chiP)])
    write(g, "chi2_total", [Float64(m.chiT)])
    write(g, "rmse_A", [Float64(m.rmseA)])
    write(g, "rmse_phi", [Float64(m.rmseP)])
    return g
end

function save_results_h5_fixed_tpk(filename_hdf5;
    timestamps,
    target_A_raw,
    target_phi_raw,
    target_A_corr,
    target_phi_corr,
    A_background_sep,
    phi_background_sep,
    A_perturbation_raw_sep,
    phi_perturbation_raw_sep,
    A_perturbation_fit_sep,
    phi_perturbation_fit_sep,
    A_seed,
    phi_seed,
    A_reco,
    phi_reco,
    pars_list::Vector{<:AbstractVector},
    initial_free_params,
    initial_full_params,
    reco_free_params,
    reco_full_params,
    free_parameter_names::Vector{String},
    full_parameter_names::Vector{String},
    t_pk_fixed,
    zdt_pk_fixed,
    k_pk,
    score_pk,
    delta_A_pk,
    delta_phi_pk,
    sigma_A_pk,
    sigma_phi_pk,
    m_seed,
    m_reco,
    reco_model::AbstractString = "unknown",
    separation_source_h5::AbstractString = "unknown",
    background_source_h5::AbstractString = "unknown",
    background_source_type::AbstractString = "unknown",
    amp_offset_db_used::Real = NaN,
    phi_offset_deg_used::Real = NaN,
)
    ts = collect(Float64.(timestamps))

    tA_raw = collect(Float64.(target_A_raw))
    tP_raw = collect(Float64.(target_phi_raw))
    tA_corr = collect(Float64.(target_A_corr))
    tP_corr = collect(Float64.(target_phi_corr))

    bgA_sep = collect(Float64.(A_background_sep))
    bgP_sep = collect(Float64.(phi_background_sep))
    pertA_raw_sep = collect(Float64.(A_perturbation_raw_sep))
    pertP_raw_sep = collect(Float64.(phi_perturbation_raw_sep))
    pertA_fit_sep = collect(Float64.(A_perturbation_fit_sep))
    pertP_fit_sep = collect(Float64.(phi_perturbation_fit_sep))

    seed_A = collect(Float64.(A_seed))
    seed_P = collect(Float64.(phi_seed))
    reco_A = collect(Float64.(A_reco))
    reco_P = collect(Float64.(phi_reco))

    init_free = collect(Float64.(initial_free_params))
    init_full = collect(Float64.(initial_full_params))
    reco_free = collect(Float64.(reco_free_params))
    reco_full = collect(Float64.(reco_full_params))

    @assert length(init_free) == 6
    @assert length(reco_free) == 6
    @assert length(init_full) == 7
    @assert length(reco_full) == 7
    @assert length(free_parameter_names) == 6
    @assert length(full_parameter_names) == 7

    true_h = collect(Float64.(pars_list[1]))
    true_beta = collect(Float64.(pars_list[2]))

    h5open(filename_hdf5, "w") do f
        # Compatibilidad plana con scripts/notebooks anteriores
        write(f, "timestamps", ts)
        write(f, "target_A", tA_raw)
        write(f, "target_phi", tP_raw)
        write(f, "target_A_corr", tA_corr)
        write(f, "target_phi_corr", tP_corr)
        write(f, "A_background_sep", bgA_sep)
        write(f, "phi_background_sep", bgP_sep)
        write(f, "A_perturbation_raw_sep", pertA_raw_sep)
        write(f, "phi_perturbation_raw_sep", pertP_raw_sep)
        write(f, "A_seed", seed_A)
        write(f, "phi_seed", seed_P)
        write(f, "A_reco", reco_A)
        write(f, "phi_reco", reco_P)
        write(f, "true_h", true_h)
        write(f, "true_beta", true_beta)
        write(f, "initial_params", init_full)
        write(f, "reco_params", reco_full)
        write(f, "initial_free_params", init_free)
        write(f, "reco_free_params", reco_free)

        # Grupos ordenados
        gtime = create_group(f, "time")
        write(gtime, "timestamps_unix", ts)

        gt = create_group(f, "targets")
        write(gt, "target_A_raw", tA_raw)
        write(gt, "target_phi_raw", tP_raw)
        write(gt, "target_A_corr", tA_corr)
        write(gt, "target_phi_corr", tP_corr)
        write(gt, "A_background_sep", bgA_sep)
        write(gt, "phi_background_sep", bgP_sep)
        write(gt, "A_perturbation_raw_sep", pertA_raw_sep)
        write(gt, "phi_perturbation_raw_sep", pertP_raw_sep)
        write(gt, "A_perturbation_fit_sep", pertA_fit_sep)
        write(gt, "phi_perturbation_fit_sep", pertP_fit_sep)
        HDF5.attributes(gt)["target_source"] = "A_real, phi_real from Whittaker/arPLS separation H5"
        HDF5.attributes(gt)["separation_source_h5"] = separation_source_h5
        HDF5.attributes(gt)["offset_rule"] = "target_A_corr=A_real+amp_offset_db_used; target_phi_corr=phi_real+phi_offset_deg_used"

        gb = create_group(f, "background")
        write(gb, "h_coeffs", true_h)
        write(gb, "beta_coeffs", true_beta)
        HDF5.attributes(gb)["background_source_h5"] = background_source_h5
        HDF5.attributes(gb)["background_source_type"] = background_source_type

        gi = create_group(f, "initial")
        write(gi, "free_params", init_free)
        write(gi, "full_params", init_full)
        write(gi, "A_model", seed_A)
        write(gi, "phi_model", seed_P)
        write(gi, "A_residual", seed_A .- tA_corr)
        write(gi, "phi_residual", seed_P .- tP_corr)
        write_metric_group!(gi, "metrics", m_seed)

        gr = create_group(f, "reconstruction")
        write(gr, "free_params", reco_free)
        write(gr, "full_params", reco_full)
        write(gr, "A_model", reco_A)
        write(gr, "phi_model", reco_P)
        write(gr, "A_residual", reco_A .- tA_corr)
        write(gr, "phi_residual", reco_P .- tP_corr)
        write_metric_group!(gr, "metrics", m_reco)

        gf = create_group(f, "fixed")
        write(gf, "t_pk_s", [Float64(t_pk_fixed)])
        write(gf, "t_pk_index", [Int(k_pk)])
        write(gf, "score_pk", collect(Float64.(score_pk)))
        write(gf, "delta_A_for_peak", collect(Float64.(delta_A_pk)))
        write(gf, "delta_phi_for_peak", collect(Float64.(delta_phi_pk)))
        write(gf, "sigma_A_pk", [Float64(sigma_A_pk)])
        write(gf, "sigma_phi_pk", [Float64(sigma_phi_pk)])
        HDF5.attributes(gf)["t_pk_utc"] = string(zdt_pk_fixed)
        HDF5.attributes(gf)["t_pk_method"] = "inherited from PREVIOUS_RESULTS_H5/fixed/t_pk_s; original score arrays copied from previous run"
        HDF5.attributes(gf)["previous_results_h5"] = PREVIOUS_RESULTS_H5
        HDF5.attributes(gf)["t_pk_reference_time"] = string(t₀)

        go = create_group(f, "optimization")
        write(go, "best_fitness", [Float64(best_fitness(result))])
        write(go, "search_lower", lower_free)
        write(go, "search_upper", upper_free)
        write(go, "relative_search_fraction", [Float64(RELATIVE_SEARCH_FRACTION)])
        write(go, "relative_search_fraction_z0", [Float64(RELATIVE_SEARCH_FRACTION_Z0)])
        write(go, "relative_search_fractions", RELATIVE_SEARCH_FRACTIONS)
        HDF5.attributes(go)["bounds_type"] = "directed_restart_relative_to_previous_reconstruction_free_params"
        HDF5.attributes(go)["bounds_rule"] = "amp,tau_u,tau_d,eps,hz: ±10%; z0_km: ±20%; seed = PREVIOUS_RESULTS_H5/reconstruction/free_params"
        HDF5.attributes(go)["free_parameter_names"] = join(free_parameter_names, ",")
        HDF5.attributes(go)["full_parameter_names"] = join(full_parameter_names, ",")
        HDF5.attributes(go)["t_pk_optimized"] = "false"
        HDF5.attributes(go)["num_dimensions"] = "6"
        HDF5.attributes(go)["method"] = "dxnes"
        HDF5.attributes(go)["previous_results_h5"] = PREVIOUS_RESULTS_H5
        HDF5.attributes(go)["restart_seed_source"] = "PREVIOUS_RESULTS_H5/reconstruction/free_params"
        HDF5.attributes(go)["restart_tpk_source"] = "PREVIOUS_RESULTS_H5/fixed/t_pk_s"

        HDF5.attributes(f)["reco_model"] = reco_model
        HDF5.attributes(f)["parameter_names"] = join(full_parameter_names, ",")
        HDF5.attributes(f)["free_parameter_names"] = join(free_parameter_names, ",")
        HDF5.attributes(f)["separation_source_h5"] = separation_source_h5
        HDF5.attributes(f)["background_source_h5"] = background_source_h5
        HDF5.attributes(f)["background_source_type"] = background_source_type
        HDF5.attributes(f)["background_family"] = "Whittaker/arPLS separation 16-22 restart15"
        HDF5.attributes(f)["old_asls_used"] = "false"
        HDF5.attributes(f)["target_source"] = "A_real, phi_real from separation H5"
        HDF5.attributes(f)["amp_offset_db_used"] = Float64(amp_offset_db_used)
        HDF5.attributes(f)["phi_offset_deg_used"] = Float64(phi_offset_deg_used)
        HDF5.attributes(f)["t_pk_optimized"] = "false"
        HDF5.attributes(f)["t_pk_fixed_s"] = Float64(t_pk_fixed)
        HDF5.attributes(f)["t_pk_fixed_utc"] = string(zdt_pk_fixed)
        HDF5.attributes(f)["bounds_type"] = "directed_restart_relative_to_previous_reconstruction_free_params"
        HDF5.attributes(f)["relative_search_fraction"] = Float64(RELATIVE_SEARCH_FRACTION)
        HDF5.attributes(f)["relative_search_fraction_z0"] = Float64(RELATIVE_SEARCH_FRACTION_Z0)
        HDF5.attributes(f)["previous_results_h5"] = PREVIOUS_RESULTS_H5
        HDF5.attributes(f)["restart_seed_source"] = "PREVIOUS_RESULTS_H5/reconstruction/free_params"
        HDF5.attributes(f)["restart_tpk_source"] = "PREVIOUS_RESULTS_H5/fixed/t_pk_s"
    end
end

jobid = get(ENV, "SLURM_JOB_ID", "nojid")
filename_hdf5 = "new-perturbation-inversion_F3_nexp_whittakerBG12976567_FIXED_TPK_restart_from15517278_rel10_z0wide20_results_$(jobid).h5"

save_results_h5_fixed_tpk(
    filename_hdf5;
    timestamps = timestamps_test,
    target_A_raw = target_A_raw,
    target_phi_raw = target_phi_raw,
    target_A_corr = target_A_corr,
    target_phi_corr = target_phi_corr,
    A_background_sep = A_background_target_sep,
    phi_background_sep = phi_background_target_sep,
    A_perturbation_raw_sep = A_perturbation_raw_target_sep,
    phi_perturbation_raw_sep = phi_perturbation_raw_target_sep,
    A_perturbation_fit_sep = A_perturbation_fit_target_sep,
    phi_perturbation_fit_sep = phi_perturbation_fit_target_sep,
    A_seed = A_seed,
    phi_seed = ϕ_seed,
    A_reco = A_reco,
    phi_reco = ϕ_reco,
    pars_list = pars_list,
    initial_free_params = initial_free_params,
    initial_full_params = initial_full_params,
    reco_free_params = reco_free_params,
    reco_full_params = reco_full_params,
    free_parameter_names = FREE_PARAMETER_NAMES,
    full_parameter_names = FULL_PARAMETER_NAMES,
    t_pk_fixed = t_pk_fixed,
    zdt_pk_fixed = zdt_pk_fixed,
    k_pk = k_pk,
    score_pk = score_pk,
    delta_A_pk = ΔA_pk,
    delta_phi_pk = Δϕ_pk,
    sigma_A_pk = σA_pk,
    sigma_phi_pk = σϕ_pk,
    m_seed = m_seed,
    m_reco = m_reco,
    reco_model = "F3_gaussiana_asimetrica_nexp_vertical_FIXED_TPK_RESTART_FROM_15517278_Z0WIDE20",
    separation_source_h5 = SEPARATION_H5,
    background_source_h5 = BACKGROUND_H5,
    background_source_type = "whittaker_arpls_sep16_22_restart15_12976567_reconstruction_h_beta_coeffs",
    amp_offset_db_used = AMP_OFFSET_DB_USED,
    phi_offset_deg_used = PHI_OFFSET_DEG_USED,
)

println("\nArchivo HDF5 guardado: ", filename_hdf5)
println("Inversión terminada: restart F3+nexp con t_pk fijo heredado y 6 parámetros libres; z0_km con ±20%, resto ±10%.")
