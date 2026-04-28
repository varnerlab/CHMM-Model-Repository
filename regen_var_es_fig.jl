# ========================================================================================= #
# regen_var_es_fig.jl
#
# Standalone re-render of the VaR/ES envelope figure (paper Figure 7) using the
# already-saved deterministic numerical summary at
#   results/diagnostics/utility/VaR_ES_Backtest.txt
# (produced by run_diagnostics.jl with SEED = 20260420).
#
# Mirrors the plotting block in run_diagnostics.jl [4.1] verbatim, with the
# clipping fix: bumped left_margin (22mm) and labels only on the left column.
# Avoids re-running the expensive CHMM/GARCH training pipeline since the seeded
# numerical values are already on disk.
# ========================================================================================= #

using Pkg; Pkg.activate(@__DIR__);
using Plots;
include(joinpath(@__DIR__, "plots_defaults.jl"));

const N_PATHS = 1000;
const TXT_PATH = joinpath(@__DIR__, "results", "diagnostics", "utility", "VaR_ES_Backtest.txt");
const OUT_DIR  = joinpath(@__DIR__, "results", "diagnostics", "utility");

# ----- Parse VaR_ES_Backtest.txt into per-model summaries ----------------------- #
# Two blocks of interest:
#   (a) Observed VaR/ES values for IS and OoS at α = 0.01, 0.05.
#   (b) Per-model rows: med [lo, hi] for IS01/IS05/OS01/OS05, both v and e.

obs = Dict{Symbol,Float64}();
models_var = Dict{String,Any}();

function parse_med_lohi(cell::AbstractString)
    # "med [lo, hi]"
    s = strip(cell);
    parts = split(s, "[");
    med = parse(Float64, strip(parts[1]));
    lohi = strip(replace(parts[2], "]" => ""));
    lo, hi = parse.(Float64, strip.(split(lohi, ",")));
    return med, lo, hi;
end

open(TXT_PATH, "r") do io
    for line in eachline(io)
        ln = strip(line);
        if startswith(ln, "IS observed VaR01")
            m = match(r"VaR01=(-?\d+\.?\d*)\s+ES01=(-?\d+\.?\d*)", ln);
            obs[:is_v01] = parse(Float64, m.captures[1]);
            obs[:is_e01] = parse(Float64, m.captures[2]);
        elseif startswith(ln, "IS observed VaR05")
            m = match(r"VaR05=(-?\d+\.?\d*)\s+ES05=(-?\d+\.?\d*)", ln);
            obs[:is_v05] = parse(Float64, m.captures[1]);
            obs[:is_e05] = parse(Float64, m.captures[2]);
        elseif startswith(ln, "OoS observed VaR01")
            m = match(r"VaR01=(-?\d+\.?\d*)\s+ES01=(-?\d+\.?\d*)", ln);
            obs[:os_v01] = parse(Float64, m.captures[1]);
            obs[:os_e01] = parse(Float64, m.captures[2]);
        elseif startswith(ln, "OoS observed VaR05")
            m = match(r"VaR05=(-?\d+\.?\d*)\s+ES05=(-?\d+\.?\d*)", ln);
            obs[:os_v05] = parse(Float64, m.captures[1]);
            obs[:os_e05] = parse(Float64, m.captures[2]);
        else
            for name in ["Bootstrap", "GARCH", "CHMM-N", "CHMM-t", "CHMM-L"]
                if startswith(ln, name)
                    cells = strip.(split(ln, "|"));
                    # cells[1]=name,
                    # cells[2..9] = IS V01, IS E01, IS V05, IS E05, OoS V01, OoS E01, OoS V05, OoS E05
                    isv01 = parse_med_lohi(cells[2]);
                    ise01 = parse_med_lohi(cells[3]);
                    isv05 = parse_med_lohi(cells[4]);
                    ise05 = parse_med_lohi(cells[5]);
                    osv01 = parse_med_lohi(cells[6]);
                    ose01 = parse_med_lohi(cells[7]);
                    osv05 = parse_med_lohi(cells[8]);
                    ose05 = parse_med_lohi(cells[9]);
                    models_var[name] = (
                        is01 = (v_med=isv01[1], v_lo=isv01[2], v_hi=isv01[3],
                                e_med=ise01[1], e_lo=ise01[2], e_hi=ise01[3]),
                        is05 = (v_med=isv05[1], v_lo=isv05[2], v_hi=isv05[3],
                                e_med=ise05[1], e_lo=ise05[2], e_hi=ise05[3]),
                        os01 = (v_med=osv01[1], v_lo=osv01[2], v_hi=osv01[3],
                                e_med=ose01[1], e_lo=ose01[2], e_hi=ose01[3]),
                        os05 = (v_med=osv05[1], v_lo=osv05[2], v_hi=osv05[3],
                                e_med=ose05[1], e_lo=ose05[2], e_hi=ose05[3]),
                    );
                    break;
                end
            end
        end
    end
end

@assert length(models_var) == 5  "Expected 5 models, got $(length(models_var))"
@assert haskey(obs, :is_v01) && haskey(obs, :os_v05)  "Observed values not parsed"

println("Parsed observed values: ", obs);
println("Parsed models: ", collect(keys(models_var)));

# ----- Render figure (mirrors run_diagnostics.jl [4.1]) ------------------------- #
var_fig = plot(layout=(2,2), size=(1100,800),
    left_margin=22Plots.mm,
    bottom_margin=14Plots.mm);

model_names = ["Bootstrap", "GARCH", "CHMM-N", "CHMM-t", "CHMM-L"];
xs = 1:length(model_names);

for (i, (tag, obs_v, obs_e, key)) in enumerate([
    ("IS VaR (0.01)",  obs[:is_v01], obs[:is_e01], :is01),
    ("IS ES  (0.01)",  obs[:is_v01], obs[:is_e01], :is01),
    ("OoS VaR (0.05)", obs[:os_v05], obs[:os_e05], :os05),
    ("OoS ES  (0.05)", obs[:os_v05], obs[:os_e05], :os05)])

    is_var = startswith(tag, "IS VaR") || startswith(tag, "OoS VaR");
    meds = [is_var ? models_var[n][key].v_med : models_var[n][key].e_med for n in model_names];
    los  = [is_var ? models_var[n][key].v_lo  : models_var[n][key].e_lo  for n in model_names];
    his  = [is_var ? models_var[n][key].v_hi  : models_var[n][key].e_hi  for n in model_names];
    obs_line = is_var ? obs_v : obs_e;

    ylab = (i == 1 || i == 3) ? "Annualized log excess growth rate" : "";
    scatter!(var_fig, xs, meds, yerror=(meds .- los, his .- meds),
        subplot=i, title=tag, ms=6, color=:navy, label="sim median [5-95]",
        xticks=(xs, model_names),
        ylabel=ylab);
    hline!(var_fig, [obs_line], subplot=i, color=:red, lw=2, ls=:dash, label="observed");
end

savefig(var_fig, joinpath(OUT_DIR, "VaR_ES_Backtest.pdf"));
savefig(var_fig, joinpath(OUT_DIR, "VaR_ES_Backtest.svg"));
println("Wrote: ", joinpath(OUT_DIR, "VaR_ES_Backtest.pdf"));
