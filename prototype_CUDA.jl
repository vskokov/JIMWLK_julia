cd(@__DIR__)
using Distributions
using Random
using GellMannMatrices
using LinearAlgebra
using CUDA
using BenchmarkTools
import CUDA.CUFFT
import FFTW

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

    K_of_k .= a2*FFTW.fft(K)
    if (i==2)
        K_of_k .= transpose(K_of_k)
    end
end


ξ = CUDA.zeros(ComplexF32,(N,N,N,N))
CUFFT.fft!(ξ,(1,2))

K_of_k = zeros(ComplexF32,(N,N))
WW_kernel!(1,K_of_k)

res = CUDA.zeros(ComplexF32,(N,N))

function convolution_kernel(ξ, res)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for idx in index:stride:N^2-1
        i = idx÷N + 1
        j = idx%N + 1

        res[i,j] = ξ[i,j,i,j]
    end
    return
end

K_of_k = CuArray(K_of_k)

kernel = @cuda launch=false convolution_kernel(ξ, res)
config = launch_configuration(kernel.fun)
threads = min(N^2, config.threads)
blocks = cld(N^2, threads)

function convolution(ξ, K_of_k, res)
    ξ .= ξ.*K_of_k
    CUFFT.ifft!(ξ,(1,2))
    kernel(ξ, res; threads, blocks)
end

@btime begin
    CUDA.@sync convolution($ξ, K_of_k, $res)
end