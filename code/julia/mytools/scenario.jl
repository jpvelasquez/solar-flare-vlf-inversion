# Selecting TX and RX stations 
tx_station2 = "NAA"#"NPM"
tx_station = "WWVB"
rx_station = "PLO"
rx_station2 = "PIU"
tx_selected = tx_table[tx_table.Prefix .== tx_station, :]
tx_selected2 = tx_table[tx_table.Prefix .== tx_station2, :]

rx_selected = rx_table[rx_table.Prefix .== rx_station, :]
tx_params = Dict(column => tx_selected[1, column] for column in names(tx_selected))
tx_params2 = Dict(column => tx_selected2[1, column] for column in names(tx_selected2))

rx_params = Dict(column => rx_selected[1, column] for column in names(rx_selected))

rx_selected2 = rx_table[rx_table.Prefix .== rx_station2, :]
rx_params2 = Dict(column => rx_selected2[1, column] for column in names(rx_selected2))


ht = FIRI.ALTITUDE
alt = 0:1e3:150e3
alt_fine = 0:1e2:150e3

# Setting TX and RX features
tx_dip = VerticalDipole()
tx = Transmitter{typeof(tx_dip)}(
    tx_params["Prefix"], tx_params["Latitude"], tx_params["Longitude"], 
    tx_dip, Frequency(tx_params["Frequency_kHz"]*1e3), tx_params["Power_kW"]*1e3
    )
tx2 = Transmitter{typeof(tx_dip)}(
    tx_params2["Prefix"], tx_params2["Latitude"], tx_params2["Longitude"],
    tx_dip, Frequency(tx_params2["Frequency_kHz"]*1e3), tx_params2["Power_kW"]*1e3
    )

# Computing distance from TX to RX in meters
d_radar = Geodesics.inverse(deg2rad.((tx_params["Longitude"], tx_params["Latitude"], rx_params["Longitude"], rx_params["Latitude"]))..., a, f)[1]
println("Geodesic Distance from $tx_station to $rx_station: $(round(d_radar/1e3, digits=2)) km")

d_radar2 = Geodesics.inverse(deg2rad.((tx_params["Longitude"], tx_params["Latitude"], rx_params2["Longitude"], rx_params2["Latitude"]))..., a, f)[1]
println("Geodesic Distance from $tx_station to $rx_station2: $(round(d_radar2/1e3, digits=2)) km")


d2_radar = Geodesics.inverse(deg2rad.((tx_params2["Longitude"], tx_params2["Latitude"], rx_params["Longitude"], rx_params["Latitude"]))..., a, f)[1]
println("Geodesic Distance from $tx_station2 to $rx_station: $(round(d_radar/1e3, digits=2)) km")

d2_radar2 = Geodesics.inverse(deg2rad.((tx_params2["Longitude"], tx_params2["Latitude"], rx_params2["Longitude"], rx_params2["Latitude"]))..., a, f)[1]
println("Geodesic Distance from $tx_station2 to $rx_station2: $(round(d_radar2/1e3, digits=2)) km")


#ranges = 0:500e3:d_radar
#ranges2 = 0:500e3:d_radar2
ranges = 0:10e3:d_radar
ranges2 = 0:10e3:d_radar2
ranges12 = 0:10e3:d2_radar
ranges22 = 0:10e3:d2_radar2


field = Fields.Ez
rx_samp = GroundSampler(ranges, field) # ATI (lat -23.1833, lon -46.6000)
rx = GroundSampler(d_radar, field)

rx_samp2 = GroundSampler(ranges2, field) # ATI (lat -23.1833, lon -46.6000)
rx2 = GroundSampler(d_radar2, field)

rx_samp12 = GroundSampler(ranges12, field) # ATI (lat -23.1833, lon -46.6000)
rx12 = GroundSampler(d2_radar, field)

rx_samp22 = GroundSampler(ranges22, field) # ATI (lat -23.1833, lon -46.6000)
rx22 = GroundSampler(d2_radar2, field)



# Building B field
Bmagnitude = 50000e-9 # [T] the field strength 25566.3 nT
Bdip = π/2            # [rad] angle from the horizontal, (+) when directed into Earth
Baz = 0               # [rad] angle from the propagation direction, (+) towards y
bfield = BField(Bmagnitude, Bdip, Baz)

# Setting ground permittivity and conductivity
# ϵ = 10
# σ = 2e-4
ground = GROUND[10] # Parameters for ocean
ground2 = GROUND[2] # Parameters for dry land
zdt = ZonedDateTime(2008, 1, 10, 10, 0, 0, tz"UTC")
zdt = astimezone(zdt, tz"UTC")
println("Initial time: $(zdt)")
# Defining temporal range and resolution
zdatetimes = datelinspace(zdt, zdt + Hour(24), Minute(5))
# Defining 24h time vector
night_start = ZonedDateTime(Date(zdt), Time(5,0,0), tz"UTC")
day_start = ZonedDateTime(Date(zdt), Time(17,0,0), tz"UTC")

zdt_night = datelinspace(night_start, night_start+Hour(5), Hour(1))
zdt_sunrise = datelinspace(zdt_night[end], day_start, Minute(5))
zdt_day = datelinspace(day_start, day_start+Hour(6), Minute(30))
# zdt_sunset = datelinspace(zdt_day[end], night_start, Minute(5))
zdt_sunset = datelinspace(zdt_day[end], zdt_day[end]+Hour(6), Minute(10))

zdt_24h = sort(unique(vcat(zdt_night, zdt_sunrise, zdt_day, zdt_sunset)))


Nwg = [1, 3, 10]
resolution = 2*Nwg .- 1
res_dict = Dict("night" => resolution[1], "day" => resolution[2], "transition" => resolution[3])
println("Scenario defined.")

ztimerange = copy(zdatetimes)
timerange = DateTime.(ztimerange);
Nt = length(timerange)
println("Datetime range: from $(Dates.format(ztimerange[1], "yyyy-mm-dd HH:MM:SS Z")) to $(Dates.format(ztimerange[end], "yyyy-mm-dd HH:MM:SS Z")) with a length of $Nt.")
