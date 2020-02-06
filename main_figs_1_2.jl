using LinearAlgebra
using Plots
using LaTeXStrings


using LightGraphs
using GraphIO
using EzXML
#using CSV
using MAT

using Distributions

using DelimitedFiles
using ColorSchemes
using Colors

include("my_funcs.jl")

# gener_sig

g = loadgraph("data/MyGraph.graphml", GraphIO.GraphML.GraphMLFormat())

#node_labels = CSV.read("data/MyGraph_clust.gml.csv")[:modularity_class]
node_labels = Int.(readdlm("data/MyGraph_clust.gml.csv",',';skipstart=1)[:,4])

cov_h0 = [1.0 0.0; 0.0 1.0]
cov_h1 = [1.0 0.9; 0.9 1.0]
sigs = gener_sig(g, node_labels, cov_h0, cov_h1; nt = 2000, n_c = 1400)

# nougat !

pdf_h0 = MvNormal(zeros(2), cov_h0)
pdf_h1 = MvNormal(zeros(2), cov_h1)
dict = [rand(pdf_h0, 40) rand(pdf_h1, 40)]

n_ref = n_test = 128
μ = 0.01
λ = 0.01
γ = .3

l_est = zeros(nv(g), size(sigs)[3] - n_ref - n_test +1)
for k in 1:nv(g)
    l_est[k, :] = nougat(sigs[k,:,:], dict, n_ref, n_test, μ, λ, γ)
end

# ARMA

poles = [ 0.9284586365913845 + 0.6691948262233165im 0.9284586365913845 - 0.6691948262233165im -0.9284586223955065 + 0.6691948202913867im -0.9284586223955065 - 0.6691948202913867im]
residues = [-0.09550841212039587 - 0.10204555134224505im -0.09550841212039587 + 0.10204555134224504im -0.023277450874456127 - 0.8479373939514138im  -0.023277450874456127 + 0.8479373939514138im ]
c = 0.6682305081233931

g_l = arma_graph(l_est, g, poles, residues, c)

# save results

file = matopen("data/simu1.mat", "w")
write(file, "l_est", l_est)
write(file, "g_l", g_l)
close(file)

# plot 1

file = matopen("data/simu1.mat", "w")
l_est = read(file, "l_est")
g_l = read(file, "g_l")
close(file)

t_gfss = [norm(g_l[:,k])^2 for k in 1:size(g_l)[2]]
t_nougat = [norm(l_est[:,k])^2 for k in 1:size(g_l)[2]]

# test

gr()
plot( 256:2000, [t_nougat t_gfss], label="")

# final

pyplot()

Plots.reset_defaults()
Plots.scalefontsizes(2)

lab1 = L"\parallel \hat{\mathbf{\ell}}_t \parallel_2^2"
lab2 = L"\parallel \hat{\mathbf{g}}_{\hat{\mathbf{\ell}}_t} \parallel _2^2"
plot( 256:2000, [t_nougat t_gfss], label = [lab1 lab2], xlab=L"t", w=1, dpi=300, legend=:topleft)
plot!([1400], seriestype = :vline, label="", w=2)
annotate!(1340, 5, text(L"t_r",18))
plot!([1529], seriestype = :vline, label="", w=2)
annotate!(1490, 5, text(L"t^\ast",18))

savefig("figs/global_2.pdf")

# plot 2 with gephi

gr()
plot(rescale(abs.(l_est[:,1272])))
plot!(rescale(abs.(g_l[:,1272])))

write_nodes_sig(abs.(l_est[:,1272]), "data/sans_filtre_2.csv")
write_nodes_sig(abs.(g_l[:,1272]), "data/avec_filtre_2.csv")
