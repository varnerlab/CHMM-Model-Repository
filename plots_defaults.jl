# ========================================================================= #
# plots_defaults.jl
#
# Project-wide Plots.jl defaults for the CHMM-Model figure pipeline.
#
# Loaded exactly once, from Include.jl, after `using Plots;`. All `plot(...)`
# and composite `plot(p1, p2, ...; layout=...)` calls inherit these settings
# unless the call site explicitly overrides a keyword.
#
# Rationale: the GR backend's default PDF MediaBox has zero margin on every
# edge. With no explicit margin, rotated y-axis titles and x-axis titles are
# placed right against the page boundary and get clipped by downstream PDF
# viewers and by pdflatex `\includegraphics`. Setting a generous project-wide
# default keeps ink inside the MediaBox with whitespace on all four sides.
#
# Per-figure overrides: pass the same keyword to an individual `plot(...)`
# call to override the default for that figure only.
#
#   plot(...; left_margin = 18Plots.mm)   # needs more than the default
# ========================================================================= #

Plots.default(
    left_margin   = 14Plots.mm,
    bottom_margin = 12Plots.mm,
    right_margin  =  6Plots.mm,
    top_margin    =  6Plots.mm,
    guidefontsize = 11,
    tickfontsize  = 10,
    legendfontsize = 9,
    titlefontsize = 12,
)
