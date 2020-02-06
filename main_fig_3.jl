using MAT
using DelimitedFiles
using Plots
using LaTeXStrings


file = matopen("data/simu2_ditrib_monte_carlo.mat")
l_est = read(file, "mc_l_est") # note that this does NOT introduce a variable ``varname`` into scope;
g_l = read(file, "mc_g_l")
close(file)

node_labels = Int.(readdlm("data/MyGraph_clust.gml.csv",',';skipstart=1)[:,4])

n_ξ = 256

# nougat core

l_est_h1 = abs.(l_est[findall(in(4), node_labels),:][:])
l_est_h0 = abs.(l_est[findall(!in(4), node_labels),:][:])

# plot(l_est_h0)
# plot!(l_est_h1)

ξ = range(0, stop=maximum(l_est_h1), length=n_ξ)

pfa_no_filter = zeros(n_ξ)
pdet_no_filter = zeros(n_ξ)
for k in 1:n_ξ
    pfa_no_filter[k] = count(x->x>ξ[k], l_est_h0)
    pdet_no_filter[k] = count(x->x>ξ[k], l_est_h1)
end

pfa_no_filter /= length(l_est_h0)
pdet_no_filter /= length(l_est_h1)

# filtered nougat core

g_l_h1 = abs.(g_l[findall(in(4), node_labels),:][:])
g_l_h0 = abs.(g_l[findall(!in(4), node_labels),:][:])

# plot(g_l_h0)
# plot!(g_l_h1)

ξ = range(0, stop=maximum(l_est_h1), length=n_ξ)

pfa_with_filter = zeros(n_ξ)
pdet_with_filter = zeros(n_ξ)
for k in 1:n_ξ
    pfa_with_filter[k] = count(x->x>ξ[k], g_l_h0)
    pdet_with_filter[k] = count(x->x>ξ[k], g_l_h1)
end

pfa_with_filter /= length(g_l_h0)
pdet_with_filter /= length(g_l_h1)

# plots

#gr()
#plot(pfa_with_filter, pdet_with_filter, xlims=(0.0, 1.0), ratio=:equal)
#plot!(pfa_no_filter, pdet_no_filter)

# final


pyplot()

Plots.reset_defaults()
Plots.scalefontsizes(1.5)

#lab1 = L"$\hat{\mathbf{\ell}}_{t}$ for $t \in t^\ast$: $C^\ast$ vs $V\backslash C^\ast$"
#lab2 = L"$\hat{\mathbf{g}}_{\hat{\mathbf{\ell}}_t}$ for $t \in t^\ast$: $C^\ast$ vs $V\backslash C^\ast$"

lab1 = L"ROC for $\hat{\mathbf{\ell}}_{t}$ when $t = t^\ast$"
lab2 = L"ROC for $\hat{\mathbf{g}}_{\hat{\mathbf{\ell}}_t}$ when $t = t^\ast$"

plot(pfa_with_filter, pdet_with_filter, label = lab2, w=2, legend=:bottomright,xlims=(0.0, 1.0), ratio=:equal)
plot!(pfa_no_filter, pdet_no_filter, label = lab1, xlabel=L"P_{fa}", ylabel=L"P_d",w=2,dpi=300)


savefig("figs/core.pdf")
