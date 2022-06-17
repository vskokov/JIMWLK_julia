using Distributions
using StaticArrays
using Random
using DelimitedFiles
using SLEEF
using JLD2
using GellMannMatrices
using LinearAlgebra
using FFTW


const mu² = 1

l = 32
N = 512
a = l/N
const Ny = 10
const m² = 0.02
const Nc=3

const variance_of_mv_noise = sqrt(mu² / (Ny * a^2))

seed=abs(rand(Int))
rng = MersenneTwister(seed);


t=gellmann(Nc,skip_identity=false)/2
t[9]=t[9]*sqrt(2/3)


function generate_rho_fft_to_momentum_space()
    rho = variance_of_mv_noise * randn(rng, Float32,(N,N))
    a^2*fft(rho)
end

function compute_field!(rhok)
# Modifies the argument to return the field
    Threads.@threads for i in 1:N
        for j in 1:N
            rhok[i,j] = rhok[i,j] / (a^2 * m² + 4.0 * sin(π*(i-1)/N)^2 + 4.0 * sin(π*(j-1)/N)^2)
            # factor of a^2 was removed to account for the normalization of ifft next
            # ifft computes sum / (lenfth of array) for each dimension
        end
    end
    ifft!(rhok)
end


function compute_local_fund_Wilson_line()
    A_arr = Array{Matrix{ComplexF64}}(undef, (N,N))
    V = Array{Matrix{ComplexF64}}(undef, (N,N))

    Threads.@threads for i in 1:N
        for j in 1:N
            A_arr[i,j] = zeros(ComplexF32,(Nc,Nc))
            V[i,j] = zeros(ComplexF32,(Nc,Nc))
        end
    end

    for b in 1:Nc^2-1

        ρ_k = generate_rho_fft_to_momentum_space()
        compute_field!(ρ_k)
        A = real.(ρ_k)

        Threads.@threads for i in 1:N
            for j in 1:N
                A_arr[i,j] .= A_arr[i,j] + A[i,j].*t[b]
            end
        end
    end

    Threads.@threads for i in 1:N
        for j in 1:N
            V[i,j] .= exp(1.0im.*A_arr[i,j])
        end
    end
    return V
end


function compute_path_ordered_fund_Wilson_line()
    V = compute_local_fund_Wilson_line()
    for i in 1:Ny-1
        display(i)
        tmp=compute_local_fund_Wilson_line()
        Threads.@threads for i in 1:N
            for j in 1:N
                V[i,j] .= V[i,j]*tmp[i,j]
            end
        end
    end
    V
end

function V_components(V)
    V_comp=zeros(ComplexF32,Nc^2)
    for b in 1:Nc^2
        V_comp[b]=2.0*tr(V*t[b])
    end
    return V_comp
end


function compute_field_of_V_components(V)
    Vc=zeros(ComplexF64, (Nc^2,N,N))

    Threads.@threads for b in 1:Nc^2
        for i in 1:N
            for j in 1:N
                Vc[b,i,j]=2.0*tr(V[i,i]*t[b])
            end
        end
    end
    return Vc
end

function FFT_Wilson_components(Vc)
    Vk=zeros(ComplexF64, (Nc^2,N,N))

    for b in 1:Nc^2
        Vk[b,:,:] .= a^2 .* fft(Vc[b,:,:])
    end

    return Vk
end

function test(V)
    display("Test GM")
    display(sum( (t[i] * t[i] for i in 1:8)) )
    display(tr(t[1] * t[2])==0)
    display(tr(t[1] * t[1])==0.5)
    display(tr(t[9] * t[9])==0.5)

    testV=V_components(V[1,1])

    testM=sum(testV[b]*t[b] for b in 1:Nc^2) .- V[1,1]

    display(testM)

    display(V[1,1]*adjoint(V[1,1]))
end

function dipole(Vk)
    Sk = zeros(ComplexF64, (N,N))
    for i in 1:N
        for j in 1:N
            Sk[i,j] = 0.5*sum(Vk[b,i,j]*conj(Vk[b,i,j]) for b in 1:Nc^2)/Nc
        end
    end
    ifft(Sk)./a^2/l^2 # accounts for impact parameter integral
end

function k2(i,j)
    return((4.0 * sin(π*(i-1)/N)^2 + 4.0 * sin(π*(j-1)/N)^2)/a^2)
end


function bin_x(S)
    Nbins=N
    Sb=zeros(Float32,Nbins)
    Nb=zeros(Float32,Nbins)
    for i in 1:N
        for j in 1:N
            r=(i-1)*a+(j-1)*a
            idx_r = floor(Int,r / (2*a))+1
            if(idx_r<=Nbins)
                Sb[idx_r]=Sb[idx_r]+real(S[i,j])
                Nb[idx_r]=Nb[idx_r]+1
            end
        end
    end

    return(Sb ./ Nb)
    end


V=compute_path_ordered_fund_Wilson_line()
test(V)


Vc=compute_field_of_V_components(V)

sum(Vc[:,1,1].*conj.(Vc[:,1,1]))

Vk=FFT_Wilson_components(Vc)

S=dipole(Vk)

S[1,1]

bin_x(S)

plot((1:N)*2a,bin_x(S))
