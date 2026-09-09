import DataFrames: DataFrame
import XLSX
#using Interpolations
#const ITP = Interpolations


const tx_table = DataFrame(XLSX.readtable("./Documentos/stations.xlsx", "tx"))
const rx_table = DataFrame(XLSX.readtable("./Documentos/stations.xlsx", "rx"))

using Printf
# using Geodesy: LLA, euclidean_distance
using Geodesics
a, f = Geodesics.EARTH_R_MAJOR_WGS84, Geodesics.F_WGS84;
import GMT


function geolinspace(P1, P2, N::Int)
    # N: Number of points between extremal points
    long1, lat1 = P1[1], P1[2]
    long2, lat2 = P2[1], P2[2]
    n = N+2

    total_distance = Geodesics.inverse(deg2rad.((long1, lat1, long2, lat2))..., a, f)[1]/1e3
    ortho = GMT.orthodrome([long1 lat1; long2 lat2], step=total_distance/(N+1), unit=:k)
    lons, lats = ortho[:,1], ortho[:,2] 
    dists = [Geodesics.inverse(deg2rad.((ortho[1,:][1], ortho[1,:][2], ortho[i,:][1], ortho[i,:][2]))..., a, f)[1] for i in 1:n]
    return lats, lons, dists
end

function geolinspace2(P1, P2,P3,N::Int)
    # N: Number of points between extremal points
    long1, lat1 = P1[1], P1[2]
    long2, lat2 = P2[1], P2[2]
    long3, lat3 = P3[1], P3[2]
    n = N+2

    total_distance = Geodesics.inverse(deg2rad.((long1, lat1, long2, lat2))..., a, f)[1]/1e3
    ortho = GMT.orthodrome([long1 lat1; long2 lat2], step=total_distance/(N+1), unit=:k)
    lons, lats = ortho[:,1], ortho[:,2]
    dists = [Geodesics.inverse(deg2rad.((ortho[1,:][1], ortho[1,:][2], ortho[i,:][1], ortho[i,:][2]))..., a, f)[1] for i in 1:n]

    total_distance3 = Geodesics.inverse(deg2rad.((long1, lat1, long3, lat3))..., a, f)[1]/1e3
    ortho3 = GMT.orthodrome([long1 lat1; long3 lat3], step=total_distance/(N+1), unit=:k)
    lons3, lats3 = ortho3[:,1], ortho3[:,2]
    dists3 = [Geodesics.inverse(deg2rad.((ortho3[1,:][1], ortho3[1,:][2], ortho3[i,:][1], ortho3[i,:][2]))..., a, f)[1] for i in 1:n]


    return lats, lons, dists, lats3, lons3, dists3
end



function geolinspace(TX::String, RX::String, N::Int, verbose::Bool=false)
    
    # dist2RX = zeros(Float64, n)
    # dist2TX = zeros(Float64, n)

    # For coordinates: North and East are (+) 
    # TX coordinates
    if TX in tx_table.Prefix
        tx_selected = tx_table[tx_table.Prefix .== TX, :]
        lat_tx = tx_selected[1,"Latitude"]
        long_tx = tx_selected[1,"Longitude"]
    else
        println("$TX must be in $(tx_table.Prefix), assigning (0°, 0°) to (lat,long)")
        lat_tx = 0.0
        long_tx = 0.0
    end

    # RX coordinates
    if RX in rx_table.Prefix
        rx_selected = rx_table[rx_table.Prefix .== RX, :]
        lat_rx = rx_selected[1,"Latitude"]
        long_rx = rx_selected[1,"Longitude"]
    else
        println("$RX must be in $(rx_table.Prefix), assigning (0°, 0°) to (lat,long)")
        lat_rx = 0.0
        long_rx = 0.0
    end

    P_tx = [long_tx, lat_tx]
    P_rx = [long_rx, lat_rx]
    lats, lons, dists = geolinspace(P_tx, P_rx, N)
    n = N+2
    if verbose
        @printf("Computing distance from TX [%s (%.1f°, %.1f°)] to RX [%s (%.1f°, %.1f°)] using %d points\n", TX, lat_tx, long_tx, RX, lat_rx, long_rx, N)
        mid_point = [sum(lons)/n, sum(lats)/n]
        GMT.coast(region=:global, xaxis=(annot=60,), yaxis=(annot=30,), proj=(name=:ortho, center=mid_point), frame=:g,
        water=:lightblue, land=:darkgreen, title="$TX - $RX propagation path")
        GMT.plot!(lons, lats, lw=1, lc=:navy, marker=:circle, size=0.2, markeredgecolor=0, markerfacecolor=:navy, show=true)
    end

    return lats, lons, dists
end
##################################################################################

function geolinspace2(TX::String, RX::String, RX2::String, N::Int, verbose::Bool=false)

    # dist2RX = zeros(Float64, n)
    # dist2TX = zeros(Float64, n)

    # For coordinates: North and East are (+) 
    # TX coordinates
    if TX in tx_table.Prefix
        tx_selected = tx_table[tx_table.Prefix .== TX, :]
        lat_tx = tx_selected[1,"Latitude"]
        long_tx = tx_selected[1,"Longitude"]
    else
        println("$TX must be in $(tx_table.Prefix), assigning (0°, 0°) to (lat,long)")
        lat_tx = 0.0
        long_tx = 0.0
    end

    # RX coordinates
    if RX in rx_table.Prefix
        rx_selected = rx_table[rx_table.Prefix .== RX, :]
        lat_rx = rx_selected[1,"Latitude"]
        long_rx = rx_selected[1,"Longitude"]
    else
        println("$RX must be in $(rx_table.Prefix), assigning (0°, 0°) to (lat,long)")
        lat_rx = 0.0
        long_rx = 0.0
    end
    # RX2 coordinates
    if RX2 in rx_table.Prefix
        rx_selected2 = rx_table[rx_table.Prefix .== RX2, :]
        lat_rx2 = rx_selected2[1,"Latitude"]
        long_rx2 = rx_selected2[1,"Longitude"]
    else
        println("$RX2 must be in $(rx_table.Prefix), assigning (0°, 0°) to (lat,long)")
        lat_rx2 = 0.0
        long_rx2 = 0.0
    end


    P_tx = [long_tx, lat_tx]
    P_rx = [long_rx, lat_rx]
    P_rx2 = [long_rx2, lat_rx2]
    lats, lons, dists = geolinspace(P_tx, P_rx, N)
    lats2, lons2, dists2 = geolinspace(P_tx, P_rx2, N)
    n = N+2
    if verbose
        @printf("Computing distance from TX [%s (%.1f°, %.1f°)] to RX [%s (%.1f°, %.1f°)] using %d points\n", TX, lat_tx, long_tx, RX, lat_rx, long_rx, N)
        mid_point = [sum(lons)/n, sum(lats)/n]
        GMT.coast(region=:global, xaxis=(annot=60,), yaxis=(annot=30,), proj=(name=:ortho, center=mid_point), frame=:g,
        water=:lightblue, land=:darkgreen, title="$TX - $RX propagation path")
        GMT.plot!(lons, lats, lw=1, lc=:navy, marker=:circle, size=0.2, markeredgecolor=0, markerfacecolor=:navy, show=true)
	mid_point2 = [sum(lons2)/n, sum(lats2)/n]
        #GMT.coast(region=:global, xaxis=(annot=60,), yaxis=(annot=30,), proj=(name=:ortho, center=mid_point2), frame=:g,
        #water=:lightblue, land=:darkgreen, title="$TX - $RX2 propagation path")
	#GMT.plot!(lons2, lats2, lw=1, lc=:navy, marker=:circle, size=0.2, markeredgecolor=0, markerfacecolor=:navy, show=true)
    end

    return lats, lons, dists, lats2, lons2, dists2
end

##################################################################################


# # Example usage
# n = 5;
# dist2TX, latitude, longitude = geolinspace("NPM", "PLO", n);
# for i in 1:n+2
#     print("Point ",i)
#     @printf(", Latitude: %.4f", latitude[i])
#     @printf(", Longitude: %.4f ", longitude[i])
#     @printf(", Distance from TX: %.4f km\n",dist2TX[i])
# end

using Dates: DateTime, Period
using TimeZones

# Method 1: Start date, step size, and number of dates
function datelinspace(start, step::Period, num_dates::Int)
    start = coalesce(start isa DateTime ? ZonedDateTime(start, tz"UTC") : start)
    stop = start + step * (num_dates)
    return collect(start:step:stop)
end

# Method 2: Start date, stop date, and step size
function datelinspace(start, stop, step::Period)
    start = coalesce(start isa DateTime ? ZonedDateTime(start, tz"UTC") : start)
    stop = coalesce(stop isa DateTime ? ZonedDateTime(stop, tz"UTC") : stop)
    return collect(start:step:stop)
end


function find_nearest(array::Any, value::Any, tol::Any=0.05)    
    differences = abs.(array .- value)
    indices = argmin(differences)
    nearest_value = array[indices]
    # Check if the nearest value is within the specified tolerance
    if differences[indices] > tol*value
        return NaN, NaN
    else
        return nearest_value, indices
    end
end



using PyCall
py"""
from pysolar.solar import *
import datetime
"""

function get_sza(lat, lon, zdt)
    zdt = astimezone(zdt, tz"UTC")
    # Computing solar zenith angle (χ) using pysolar
    py"""
    date = datetime.datetime($(year(zdt)), $(month(zdt)), $(day(zdt)), $(hour(zdt)), $(minute(zdt)), $(second(zdt)), tzinfo=datetime.timezone.utc)
    solar_altitude = get_altitude($(lat), $(lon), date) 
    solar_zenith = 90 - solar_altitude
    """
    χ = py"solar_zenith"
    return χ 
end

const pycall_lock = ReentrantLock()

function get_sza_threadsafe(lat, lon, zdt)
    lock(pycall_lock) do
        # Asegurarse de que la zona horaria sea UTC
        zdt = astimezone(zdt, tz"UTC")
        # Llamada a PyCall de forma segura
        py"""
        date = datetime.datetime($(year(zdt)), $(month(zdt)), $(day(zdt)), $(hour(zdt)), $(minute(zdt)), $(second(zdt)), tzinfo=datetime.timezone.utc)
        solar_altitude = get_altitude($(lat), $(lon), date)
        solar_zenith = 90 - solar_altitude
        """
        χ = py"solar_zenith"
        return χ
    end
end
#=
function nefiri(; lat, lon, zdt::ZonedDateTime,
                ht::AbstractRange=55e3:1e3:110e3,
                findex::Real=75.0)

  Npoints  = length(lat)
  Nheights = length(ht)
  ne       = zeros(Nheights, Npoints)
  zdt      = astimezone(zdt, tz"UTC")
  χ_range  = get_sza(lat, lon, zdt)

  # Este es el rango original de firi:
  default_ht = 55e3:1e3:110e3

  for i in 1:Npoints
    sza_i       = min(χ_range[i], 130)
    ne_vert     = firi(sza_i, abs(lat[i]),
                       f10_7=findex,
                       month=month(zdt))
    # <-- aquí pasas los 3 argumentos:
    ne[:, i] = FIRI.extrapolate(default_ht, ne_vert, ht)
  end

  return ne
end
=#

##=
function nefiri(;lat::Any, lon::Any, zdt::ZonedDateTime, ht::Any=0:1e3:150e3, findex::Any=75.0)
    Npoints = (size(lat)==()) ? 1 : size(lat)[1]
    Nheights = size(ht)[1]
    ne = zeros(Nheights, Npoints)
    zdt = astimezone(zdt, tz"UTC")
    χ_range = get_sza(lat, lon, zdt)
    # rango original donde firi devuelve valores:
    #default_ht = 55e3:1e3:110e3 ##solo para probar

    for i=1:Npoints
        latᵢ, lonᵢ = lat[i], lon[i]
        # Compute the electron density using the 'firi' function from FIRI
        # default height array is 55 to 110 km
        sza = copy(χ_range[i])
        if χ_range[i]>130
            sza = 130
        end
        ne_vertical = firi(sza, abs(latᵢ), f10_7=findex, month=month(zdt))
        # Extrapolate the electron density
        ne_vertical_ex = FIRI.extrapolate(ne_vertical, ht)
	#ne_vertical_ex = FIRI.extrapolate(ne_vertical,collect(ht))
        ne[:,i] = ne_vertical_ex
    end
    return ne
end
##

function nefiri_fine(;lat::Any, lon::Any, zdt::ZonedDateTime, ht_fine::Any=0:500:150e3, findex::Any=75.0)
    Npoints = (size(lat)==()) ? 1 : size(lat)[1]
    Nheights = size(ht_fine)[1]
    ne = zeros(Nheights, Npoints)
    zdt = astimezone(zdt, tz"UTC")
    χ_range = get_sza(lat, lon, zdt)

    for i=1:Npoints
        latᵢ, lonᵢ = lat[i], lon[i]
        # Compute the electron density using the 'firi' function from FIRI
        # default height array is 55 to 110 km
        sza = copy(χ_range[i])
        if χ_range[i]>130
            sza = 130
        end
        #println("Función: punto 1.")
        ne_vertical = firi(sza, abs(latᵢ), f10_7=findex, month=month(zdt))
        #println("Función: punto 2.")
        # Extrapolate the electron density
        ne_vertical_ex = FIRI.extrapolate(ne_vertical, ht_fine)
        #println("Función: punto 3.")
        ne[:,i] = ne_vertical_ex
        #println("Función: punto 4.")
    end
    return ne
end

function get_res_given_sza(zdt, res=[1, 3, 10])
    zdt = astimezone(zdt, tz"UTC")
    lats_aux, lons_aux, dists_aux = geolinspace(tx_station, rx_station, 50);
    χ_range = get_sza(lats_aux, lons_aux, zdt)
    
    if all(χ_range .> 99) # All angles are above 90° (TX to RX section at nighttime)
        resolution = res[1]
        # println("nighttime")
    elseif all(χ_range .< 90) # All angles are below 90° (TX to RX section at daytime)
        resolution = res[2]
        # println("daytime")
    else # Angles are on both sides of 90° (transition between day and night)
        resolution = res[3]
        # println("transition")
    end
    return resolution
end

χ_data = [0, 25, 50, 60, 70, 80, 90, (90.0+91.8)/2, (91.8+93.6)/2, (93.6+95.4)/2, (95.4+97.2)/2, (97.2+99.0)/2, 100, 110, 120, 130, 150, 180]
β_data = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.33, 0.37, 0.40, 0.43, 0.47, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
h_data = [74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 74.0, 76.2, 78.3, 80.5, 82.7, 84.8, 87.0, 87.0, 87.0, 87.0, 87.0, 87.0]

h_day, h_night = 74, 82
β_day, β_night = 0.3, 0.5 
a₁, b₁ = 90, 100
s₁, x₁ = 0.7, 95
sigmoid(x::Real, x₀::Real, s::Real) = 1.0 / (1.0 + exp(-s*(x-x₀)))
h_gasdia(χ, χ₀=x₁, s=s₁, h_day=74, h_night=82) = (h_night-h_day)*sigmoid(χ, χ₀, s) + h_day
β_gasdia(χ, χ₀=x₁, s=s₁, β_day=0.3, β_night=0.5) = (β_night-β_day)*sigmoid(χ, χ₀, s) + β_day

h_table = DataFrame(XLSX.readtable("./Documentos/coefficients.xlsx", "h"))
β_table = DataFrame(XLSX.readtable("./Documentos/coefficients.xlsx", "b"))
h_coeff = collect(h_table[1,2:end])
β_coeff = collect(β_table[1,2:end])
h_alb(χ) = sum(h_coeff[n] * deg2rad(χ)^(n-1) for n in eachindex(h_coeff))
β_alb(χ) = sum(β_coeff[n] * deg2rad(χ)^(n-1) for n in eachindex(β_coeff))
h(χ) = (0<=χ<=93.2) ? h_alb(χ) : h_gasdia(χ, h_alb(91.15), 87)
β(χ) = (0<=χ<=67.9) ? β_alb(χ) : β_gasdia(χ, β_alb(67.9), 0.5)

function sza_transition(χ_day=a₁, χ_night=b₁)
    a₂, b₂ = χ_day, χ_night # New day, nigth limits
    s = s₁*(b₁-a₁)/(b₂-a₂)
    χ_center = a₂ + (b₂-a₂)*(x₁-a₁)/(b₁-a₁) 
    return χ_center, s
end

function newait(sza::Any,  
                ht::Any=0:1e3:150e3,
                method=nothing,
                h_day=74, h_night=82,
                β_day=0.2, β_night=0.5,
                χ_day=a₁, χ_night=b₁)
    Np = (size(sza)==()) ? 1 : size(sza)[1]
    Nalt = length(ht)
    ne = zeros(Nalt, Np)
    χ₀, s₂ = sza_transition(χ_day, χ_night)
    if method == "gasdia"
        h_vec = h_gasdia.(sza, χ₀, s₂, h_day, h_night)
        β_vec = β_gasdia.(sza, χ₀, s₂, β_day, β_night)
    elseif method in "alberto"
        h_vec = h_alb.(sza)
        β_vec = β_alb.(sza)
    elseif isnothing(method)
        h_vec = h.(sza)
        β_vec = β.(sza)
    else
        error("Method not recognized. Only valid: gasdia, alberto, or nothing")
    end

    for i in 1:Np
        h_i, β_i = h_vec[i], β_vec[i]
        ne_z = waitprofile.(ht, h_i, β_i)
        ne[:,i] = ne_z
    end
    return ne
end

function newait(lat::Any, lon::Any, zdt::ZonedDateTime, ht::Any=0:1e3:150e3, 
                method::Union{String, Nothing}=nothing,
                h_day=74, h_night=82,
                β_day=0.2, β_night=0.5,
                χ_day=a₁, χ_night=b₁)
    zdt = astimezone(zdt, tz"UTC")
    sza = get_sza.(lat, lon, zdt)
    ne = newait(sza, ht, method, h_day, h_night, β_day, β_night, χ_day, χ_night)
    return ne
end

function propagation_path(tx_station, rx_station, time_i, resolution, χ_day=a₁, χ_night=b₁)
    latitudes, longitudes, distances, sza = [], [], [], []
    actual_phases = []
    finner = 500
    make_finner = true
    lats_fine = lons_fine = dists_fine = sza_fine = []
    while make_finner
        # We get a fine vector of sza along propagation path at time_i
        lats_fine, lons_fine, dists_fine = geolinspace(tx_station, rx_station, finner)
        sza_fine = get_sza(lats_fine, lons_fine, time_i)
        δχ = (χ_night-χ_day)/4
        χ_transition_lower = χ_day-δχ
        χ_transition_upper = χ_night+δχ
        # We identify day, night and transition subintervals from sza
        ind_day = findall(x -> x < χ_transition_lower, sza_fine)
        ind_transition = findall(x -> χ_transition_lower <= x <= χ_transition_upper, sza_fine)
        ind_night = findall(x -> x > χ_transition_upper, sza_fine);
        phases = Dict("day"=> ind_day, "transition"=> ind_transition, "night"=> ind_night)
        
        actual_phases = []
        for (key, vec) in phases
            if (length(vec) == 1)
                finner *= 2
                make_finner = true
                break
            else
                make_finner = false
            end
            if ~isempty(vec)
                push!(actual_phases, key)
            end
        end
    end
    # Keep only non-empty phases and sort them by their first index
    phases = Dict(phase => phases[phase] for phase in actual_phases)
    phases_aux = sort(collect(phases), by = x -> x[2][1])
    phases_sorted_keys = map(x -> x[1], phases_aux)
    
    # We calculate segmentation distances which resolution depends on subinterval phase
    i = 1
    lats_end = 0
    lons_end = 0
    for phase in phases_sorted_keys
        # println("Phase: $phase")
        ind0_offset = i == 1 ? 0 : 1
        ind_phase = phases[phase]
        # println(ind_phase[1]-ind0_offset, " ", ind_phase[end])
        lats, lons, dists = geolinspace([lons_fine[ind_phase[1]-ind0_offset], lats_fine[ind_phase[1]-ind0_offset]], [lons_fine[ind_phase[end]], lats_fine[ind_phase[end]]], resolution[phase])
        dists .+= Geodesics.surface_distance(tx_params["Longitude"], tx_params["Latitude"], lons_fine[ind_phase[1]-ind0_offset], lats_fine[ind_phase[1]-ind0_offset], a) # a: Earth's radius
        append!(latitudes, lats[1:end-1])
        append!(longitudes, lons[1:end-1])
        append!(distances, dists[1:end-1])
        append!(sza, get_sza(lats[1:end-1], lons[1:end-1], time_i))
        lats_end = copy(lats[end])
        lons_end = copy(lons[end])
        i += 1
    end
    append!(distances, d_radar)
    append!(latitudes, lats_end)
    append!(longitudes, lons_end)
    # append!(sza, get_sza(lats_fine[end], lons_fine[end], time_i))
    append!(sza, get_sza(lats_end, lons_end, time_i))

    return [latitudes, longitudes, distances, sza], [dists_fine, sza_fine],phases_sorted_keys
end

# using LongwaveModePropagator: unwrap!
function unwrap(x)
	for ec in eachcol(x)
        v = first(ec)  # need to define v at this scope
        setv = true
        for k in eachindex(ec)
            if setv
                v = ec[k]
                isfinite(v) ? setv = false : setv = true
            end
            if !setv
                ec[k] = v = v + rem2pi(ec[k]-v, RoundNearest)
            end
        end
    end
	return x
end

plot_kwargs = Dict(
    "Axis" => (; titlesize=18, xlabelsize=16, ylabelsize=16, xminorticksvisible=true),
    "Lines" => (; linewidth=2)
)
