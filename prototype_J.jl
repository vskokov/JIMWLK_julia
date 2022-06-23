cd(@__DIR__)
using Distributions
using Random
using GellMannMatrices
using LinearAlgebra
using BenchmarkTools
using FFTW

const l = 32
const N = 128
const a = Float32(l/N)
const a2 = a^2
const Nc=3

seed=abs(rand(Int))
rng = MersenneTwister(seed);

function K_x(x,y)
    @fastmath r2 = (sin(x*π/N)^2 + sin(y*π/N)^2)*(l/π)^2
    @fastmath x = sin(x*2π/N)*l/(2π)
    return x/(r2+1e-16)
end

function WW_kernel!(i,K_of_k)
    x = collect(0:N-1)
    y = copy(x)
    K = map(Base.splat(K_x), Iterators.product(x, y))

    K_of_k .= a2*fft(K)
    if (i==2)
        K_of_k .= transpose(K_of_k)
    end
end


ξ = zeros(ComplexF32,(N,N,N,N))
fft!(ξ,(1,2))

K_of_k = zeros(ComplexF32,(N,N))
WW_kernel!(1,K_of_k)

res = zeros(ComplexF32,(N,N))

function convolution_kernel(ξ, res)
    Threads.@threads for idx in index:stride:N^2-1
        i = idx÷N + 1
        j = idx%N + 1

        res[i,j] = ξ[i,j,i,j]
    end
end

function convolution(ξ, K_of_k, res)
    ξ .= ξ.*K_of_k
    ifft!(ξ,(1,2))
    convolution_kernel(ξ, res)
end