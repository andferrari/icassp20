"""
    comp_κ(x, dict_x, γ; bias = false)

    computes the vector of k(x, x_i^{dict})
"""
function comp_κ(x, dict_x, γ)
    (k, l) = size(dict_x)
    κ = [exp(-norm(x-dict_x[:,m])^2/(2γ)) for m in 1:l]
end

function NormalizedLaplacian(g)
    adjmat = LightGraphs.LinAlg.CombinatorialAdjacency(adjacency_matrix(g))
    I - Diagonal(adjmat.D.^(-1/2))*(adjmat.A)*Diagonal(adjmat.D.^(-1/2))
end

function gener_sig(g, node_labels, cov_h0, cov_h1; nt = 1000, n_c = 600)

    pdf_h0 = MvNormal(zeros(2), cov_h0)
    pdf_h1 = MvNormal(zeros(2), cov_h1)

    sigs = zeros(nv(g), 2, nt)

    for k in 1:nv(g)
        sigs[k, :, :] .= rand(pdf_h0, nt)
    end

    for k in findall(in(4), node_labels)
        sigs[k, :, n_c:end] .= rand(pdf_h1, nt - n_c + 1)
    end

    return sigs
end


"""
    nougat(x, dict, n_ref, n_test, μ, λ, γ)

    computes nougat
"""
function nougat(x, dict, n_ref, n_test, μ, λ, γ)

    (k,l) = size(dict)
    n_iter = size(x)[2] - n_ref - n_test
    log_like = zeros(n_iter+1)
    θ = zeros(l)

    # # compute initial H_n and h_n

    κ_n = comp_κ(x[:, n_ref+n_test], dict, γ)
    log_like[1] = log(dot(κ_n , θ) + 1)

    H_nr = zeros(l,l)
    h_nr = zeros(l)
    for m in 1:n_ref
        κ = comp_κ(x[:,m], dict, γ)
        H_nr += κ*κ'
        h_nr += κ
    end
    H_nr *= 1/n_ref
    h_nr *= 1/n_ref

    h_nt = zeros(l)
    for m in n_ref + 1:n_ref + n_test
        κ = comp_κ(x[:,m], dict, γ)
        h_nt += κ
    end
    h_nt *= 1/n_test

    for n in 1: n_iter-1

        θ = θ - μ*((H_nr + λ*I)*θ + (h_nr - h_nt))

        κ_n = comp_κ(x[:, n_ref+n_test + n], dict, γ)
        log_like[n+1] = log(dot(κ_n , θ) + 1)


        # update H_nr, h_nr  and h_nt the fast way

        um_r = comp_κ(x[:, n], dict, γ)
        up_r = comp_κ(x[:, n+n_ref], dict, γ)

        H_nr += (up_r*up_r' - um_r*um_r')/n_ref
        h_nr += (up_r - um_r)/n_ref

        um_t = comp_κ(x[:, n+n_ref], dict, γ)
        up_t = comp_κ(x[:, n+n_ref+n_test], dict, γ)
        h_nt += ( up_t - um_t)/n_test

    end

    θ = θ - μ*((H_nr + λ*I)*θ + (h_nr - h_nt))

    κ_n = comp_κ(x[:, n_ref + n_test + n_iter ], dict, γ)
    log_like[n_iter+1] = log(dot(κ_n, θ) + 1)

    return log_like
end


function arma_graph(x, g, p, r, c)

    L = NormalizedLaplacian(g)
    L = L - I

    ψ = ones(length(p))'./p
    φ = r.*ψ

    (n_v,n_t) = size(x)
    g = zeros(n_v, n_t)

    v = zeros(length(ψ),n_v)
    for i in 1:length(ψ)
        v[i,:] = φ[i].re*x[:,1]
    end
    g[:,1] = sum(v,dims=1)[1,:] + c*x[:,1]

    Rev = v
    Imv = zeros(length(ψ), n_v)
    Rev2 = zeros(length(ψ), n_v)
    Imv2 = zeros(length(ψ), n_v)

    for k in 2:n_t
        for j in 1:length(ψ)
            Rev2[j,:] = (ψ[j].re)*L*Rev[j,:] - (ψ[j].im)*L*Imv[j,:] + (φ[j].re)*x[:,k]
            Imv2[j,:] = (ψ[j].im)*L*Rev[j,:] + (ψ[j].re)*L*Imv[j,:] + (φ[j].im)*x[:,k]
        end

        g[:,k] = sum(Rev2, dims=1)[1,:] + c*x[:,k]

        Rev = copy(Rev2)
        Imv = copy(Imv2)

    end
    return g

end

function rescale(x)
    if norm(x) == 0
        x_resc = x
    elseif minimum(x) == maximum(x)
        x_resc = normalize(x)
    else
        x_resc = (x .- minimum(x))./(maximum(x)-minimum(x))
    end
    x_resc
end

function write_nodes_sig(vals, path)

    palette = ColorSchemes.viridis
    vals_n = rescale(vals)
    n_vertex = length(vals)

    nodes = ["n"*string(i-1) for i in 1:n_vertex]
    colors_sig = ["#"*hex((get(palette, vals_n[i]))) for i in 1:n_vertex]

    writedlm(path, ["Id" "sig" "Color"; hcat(nodes,vals,colors_sig)], ',')

end
