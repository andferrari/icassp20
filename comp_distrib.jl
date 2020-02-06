using SharedArrays

@everywhere using LinearAlgebra
@everywhere using LightGraphs
@everywhere using Distributions
@everywhere include("/home/ferrari/ICASSP20/simu/my_funcs.jl")

using GraphIO
using EzXML
using MAT
using DelimitedFiles

## all parameters !

# graph
g = loadgraph("data/MyGraph.graphml", GraphIO.GraphML.GraphMLFormat())

node_labels = Int.(readdlm("data/MyGraph_clust.gml.csv",',';skipstart=1)[:,4])

# signals
cov_h0 = [1.0 0.0; 0.0 1.0]
cov_h1 = [1.0 0.8; 0.8 1.0]

pdf_h0 = MvNormal(zeros(2), cov_h0)
pdf_h1 = MvNormal(zeros(2), cov_h1)
dict = [rand(pdf_h0, 40) rand(pdf_h1, 40)]

# nougat

n_ref = n_test = 128
μ = 0.01
λ = 0.01
γ = .3

# filter

poles = [ 0.9284586365913845 + 0.6691948262233165im 0.9284586365913845 - 0.6691948262233165im -0.9284586223955065 + 0.6691948202913867im -0.9284586223955065 - 0.6691948202913867im]
residues = [-0.09550841212039587 - 0.10204555134224505im -0.09550841212039587 + 0.10204555134224504im -0.023277450874456127 - 0.8479373939514138im  -0.023277450874456127 + 0.8479373939514138im ]
c = 0.6682305081233931

# Monte Carlo

real_max = 1000

l_est_t_star = SharedArray{Float64,2}((nv(g), real_max))
g_l_t_star = SharedArray{Float64,2}((nv(g), real_max))
t_star = 1272

@sync @distributed for n_real in 1:real_max
    sigs = gener_sig(g, node_labels, cov_h0, cov_h1; nt = 1530, n_c = 1400)
    l_est = zeros(nv(g), size(sigs)[3] - n_ref - n_test +1)
    for k in 1:nv(g)
        l_est[k, :] = nougat(sigs[k,:,:], dict, n_ref, n_test, μ, λ, γ)
    end
    g_l = arma_graph(l_est, g, poles, residues, c)

    l_est_t_star[:,n_real] = l_est[:, t_star]
    g_l_t_star[:,n_real] = g_l[:,t_star]
end

mc_l_est = Array(l_est_t_star)
mc_g_l = Array(g_l_t_star)

# save results

#file = matopen("data/simu2_ditrib_monte_carlo.mat", "w")
#write(file, "mc_l_est", mc_l_est)
#write(file, "mc_g_l", mc_g_l)
#close(file)
