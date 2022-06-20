cd(@__DIR__)
using Distributions
using StaticArrays
using Random
using DelimitedFiles
using SLEEF
using JLD2
using GellMannMatrices
using LinearAlgebra
using FFTW
using Printf
using Plots

const mu² = 1.0

const l = 32
const N = 512
const a = l/N
const Ny = 10
const m² = 0.2^2
const Nc=3

const variance_of_mv_noise = sqrt(mu² / (Ny * a^2))

seed=abs(rand(Int))
rng = MersenneTwister(seed);


const t=gellmann(Nc,skip_identity=false)/2
t[9]=t[9]*sqrt(2/3)


function generate_rho_fft_to_momentum_space()
    rho = variance_of_mv_noise * randn(rng, Float32,(N,N))
    fft(rho)
end

function compute_field!(rhok)
# Modifies the argument to return the field
    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds rhok[i,j] = a^2*rhok[i,j] / (a^2 * m² + 4.0 * sin(π*(i-1)/N)^2 + 4.0 * sin(π*(j-1)/N)^2)
            # factor of a^2 was removed to account for the normalization of ifft next
            # ifft computes sum / (lenfth of array) for each dimension
        end
    end
    #rhok[1,1] = 0.0im
    ifft!(rhok)
end


function compute_local_fund_Wilson_line()
    A_arr = Array{Matrix{ComplexF64}}(undef, (N,N))
    V = Array{Matrix{ComplexF64}}(undef, (N,N))

    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds A_arr[i,j] = zeros(ComplexF32,(Nc,Nc))
            @inbounds V[i,j] = zeros(ComplexF32,(Nc,Nc))
        end
    end

    for b in 1:Nc^2-1

        ρ_k = generate_rho_fft_to_momentum_space()
        compute_field!(ρ_k)
        A = real.(ρ_k)

        Threads.@threads for i in 1:N
            for j in 1:N
                @inbounds A_arr[i,j] = A_arr[i,j] + A[i,j]*t[b]
            end
        end
    end

    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds V[i,j] .= exp(1.0im.*A_arr[i,j])
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
                @inbounds V[i,j] = V[i,j]*tmp[i,j]
            end
        end
    end
    V
end

function V_components(V)
    V_comp=zeros(ComplexF32,Nc^2)
    for b in 1:Nc^2
        @inbounds V_comp[b]=2.0*tr(V*t[b])
    end
    return V_comp
end


function compute_field_of_V_components(V)
    Vc=zeros(ComplexF64, (Nc^2,N,N))

    Threads.@threads for b in 1:Nc^2
        for i in 1:N
            for j in 1:N
                @inbounds Vc[b,i,j]=2.0*tr(V[i,j]*t[b])
            end
        end
    end
    return Vc
end

function FFT_Wilson_components(Vc)
    Vk=zeros(ComplexF64, (Nc^2,N,N))

    Threads.@threads for b in 1:Nc^2
        @inbounds Vk[b,:,:] .= a^2 .* fft(Vc[b,:,:])
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
    Threads.@threads for i in 1:N
        for j in 1:N
            Sk[i,j] = 0.5*sum(Vk[b,i,j]*conj(Vk[b,i,j]) for b in 1:Nc^2)/Nc
        end
    end
    return(ifft(Sk)./a^2/l^2) # accounts for impact parameter integral
end

function k2(i,j)
    return((4.0 * sin(π*(i-1)/N)^2 + 4.0 * sin(π*(j-1)/N)^2)/a^2)
end


function bin_x(S)
    Nbins=N÷2
    Sb=zeros(Float32,Nbins)
    Nb=zeros(Float32,Nbins)
    for i in 1:N÷2
        for j in 1:N÷2
            r=sqrt(((i-1)*a)^2+((j-1)*a)^2)
            idx_r = floor(Int,r / (2*a))+1
            if (idx_r<=Nbins)
                Sb[idx_r]=Sb[idx_r]+real(S[i,j])
                Nb[idx_r]=Nb[idx_r]+1
            end
        end
    end
    return(Sb ./ (1e-16.+Nb))
end


Sb=zeros(N÷2)

#open("out.dat","w") do io
    for event in 1:10

        V=compute_path_ordered_fund_Wilson_line()
        test(V)

        Vc=compute_field_of_V_components(V)
        sum(Vc[:,1,1].*conj.(Vc[:,1,1]))
        Vk=FFT_Wilson_components(Vc)
        S=dipole(Vk)
        S[1,1]
        Sb .= Sb .+ bin_x(S)

        #for kx in 1:N
        #    Printf.@printf(io, " %f", Sb[kx])
        #end
        #Printf.@printf(io, "\n")
    end
#end
x=collect(1:N÷2)*a*2
plot(x,Sb/10)
#plot!(x,exp.(-x.^2/60 .*log.(exp(1.0) .+   200.0./x)))

dataCpp=readdlm("/Users/vskokov/Dropbox/Projects/2022/MV_Jack_X_check_Julia_x_check/out.dat",' ')
plot!(dataCpp[:,1],dataCpp[:,2])

dataCpp=readdlm("/Users/vskokov/Dropbox/Projects/2022/MV_Jack_X_check_Julia_x_check/out1.dat",' ')
plot!(dataCpp[:,1],dataCpp[:,2])

dataCpp=readdlm("/Users/vskokov/Dropbox/Projects/2022/MV_Jack_X_check_Julia_x_check/out2.dat",' ')
plot!(dataCpp[:,1],dataCpp[:,2])

dataCpp=readdlm("/Users/vskokov/Dropbox/Projects/2022/MV_Jack_X_check_Julia_x_check/out3.dat",' ')
plot!(dataCpp[:,1],dataCpp[:,2])

dataCpp=readdlm("/Users/vskokov/Dropbox/Projects/2022/MV_Jack_X_check_Julia_x_check/out4.dat",' ')
plot!(dataCpp[:,1],dataCpp[:,2])
