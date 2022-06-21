cd(@__DIR__)
using Distributions
using StaticArrays
using Random
using GellMannMatrices
using LinearAlgebra
using FFTW
using Printf

const mu² = 1.0

const l = 32
const N = 128*2
const a = l/N
const a2=a^2
const Ny = 50
const m² = 0.001^2
const Nc=3
const alpha_fc = 1 # Y is measured in alpha_fc

const variance_of_mv_noise = sqrt(mu² / (Ny * a^2))


ID=ARGS[1]

seed=abs(rand(Int))
rng = MersenneTwister(seed);

t=gellmann(Nc,skip_identity=false)/2
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
    rhok[1,1] = 0.0im
    ifft!(rhok)
end


function compute_local_fund_Wilson_line()
    A_arr = Array{Matrix{ComplexF32}}(undef, (N,N))
    V = zeros(ComplexF32, (N,N,Nc,Nc))

    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds A_arr[i,j] = zeros(ComplexF32,(Nc,Nc))
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
            @inbounds V_ij=@view V[i,j,:,:]
            @inbounds V_ij .= exp(1.0im.*A_arr[i,j])
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
                @inbounds V_ij= @view V[i,j,:,:]
                @inbounds tmp_ij= @view tmp[i,j,:,:]
                @inbounds V_ij .= V_ij*tmp_ij
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
    Vc=zeros(ComplexF32, (Nc^2,N,N))

    Threads.@threads for b in 1:Nc^2
        for i in 1:N
            for j in 1:N
                @inbounds V_ij=@view V[i,j,:,:]
                @inbounds Vc[b,i,j]=2.0*tr(V_ij*t[b])
            end
        end
    end
    return Vc
end

function FFT_Wilson_components(Vc)
    Vk=zeros(ComplexF32, (Nc^2,N,N))

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

    testV=V_components(V[1,1,:,:])

    testM=sum(testV[b]*t[b] for b in 1:Nc^2) .- V[1,1,:,:]

    display(testM)

    display(V[1,1,:,:]*adjoint(V[1,1,:,:]))
end

function dipole(Vk)
    Sk = zeros(ComplexF32, (N,N))
    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds Sk[i,j] = 0.5*sum(Vk[b,i,j]*conj(Vk[b,i,j]) for b in 1:Nc^2)/Nc
        end
    end
    return(ifft(Sk)./a^2/l^2) # accounts for impact parameter integral
end

function k2(i,j)
    return((4.0 * sin(π*(i-1)/N)^2 + 4.0 * sin(π*(j-1)/N)^2)/a^2)
end


function bin_x(S)
    Nbins=N÷2
    step=2a
    Sb=zeros(Float32,Nbins)
    Nb=zeros(Float32,Nbins)
    for i in 1:N÷2
        for j in 1:N÷2
            r=sqrt(((i-1)*a)^2+((j-1)*a)^2)
            idx_r = floor(Int,r / (step))+1
            if (idx_r<=Nbins)
                @inbounds Sb[idx_r]=Sb[idx_r]+real(S[i,j])
                @inbounds Nb[idx_r]=Nb[idx_r]+1
            end
        end
    end
    return(collect(1:Nbins)*step,Sb ./ (1e-16.+Nb))
end

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

function generate_noise_Fourier_Space!(ξ,ξ_k,ξ_c)
    randn!(rng, ξ_c)

    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds ξ_1=@view ξ[1,i,j,:,:]
            @inbounds ξ_1 .= sum(ξ_c[1,i,j,b]*t[b] for b in 1:Nc^2-1)
            @inbounds ξ_2=@view ξ[2,i,j,:,:]
            @inbounds ξ_2 .= sum(ξ_c[2,i,j,b]*t[b] for b in 1:Nc^2-1)
        end
    end

    ξ_k .= fft(ξ,(2,3))
    ξ_k .= ξ_k*a # a^2/a ; 1/a is from the noise noise correlator
end


function convolution!(K, ξ_k, ξ_out)
    @inbounds sub_ξ_1 = @view ξ_k[1,:,:,:,:]
    @inbounds sub_K_1 = @view K[1,:,:]
    @inbounds sub_ξ_2 = @view ξ_k[2,:,:,:,:]
    @inbounds sub_K_2 = @view K[2,:,:]
    @fastmath @inbounds ξ_out .= (sub_ξ_1 .* sub_K_1 .+ sub_ξ_2 .* sub_K_2)/a2
    ifft!(ξ_out,(1,2))
end


function rotated_noise(ξ,ξ_R_k,V)
    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds V_ij = @view V[i,j,:,:]
            aV = adjoint(V_ij)
            for p in 1:2
                @inbounds ξ_R = @view ξ_R_k[p,i,j,:,:]
                @fastmath @inbounds ξ_R .= aV*ξ[p,i,j,:,:]*V_ij
            end
        end
    end
    fft!(ξ_R_k,(2,3))
    ξ_R_k .= ξ_R_k*a
end


function exp_Left(ξ_k, K, ξ_out, exp_out, ΔY)
    convolution!(K, ξ_k, ξ_out)
    pref=sqrt(alpha_fc*ΔY)/π
    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds Exp = @view exp_out[i,j,:,:]
            @inbounds ξ_ij = @view ξ_out[i,j,:,:]
            @fastmath Exp .= exp(-pref*1.0im*ξ_ij)
        end
    end
end

function exp_Right(ξ_k, K, ξ_out, exp_out, ΔY)
    convolution!(K, ξ_k, ξ_out)
    pref=sqrt(alpha_fc*ΔY)/π
    Threads.@threads for i in 1:N
        for j in 1:N
            @inbounds Exp = @view exp_out[i,j,:,:]
            @inbounds ξ_ij = @view ξ_out[i,j,:,:]
            @fastmath Exp .= exp(pref*1.0im*ξ_ij)
        end
    end
end

function Qs_of_S(r,S)
    Ssat = exp(-0.5)
    j=2
    for i in 2:length(S)
        if (S[i-1]-Ssat)*(S[i]-Ssat)<0
            j=i
            continue
        end
    end
    r = r[j] + (r[j-1]-r[j])/(S[j-1]-S[j])*(Ssat-S[j])
    return sqrt(2.0)/r
end


function observables(io,Y,V)

    Vc=compute_field_of_V_components(V)
    Vk=FFT_Wilson_components(Vc)
    S=dipole(Vk)
    (r,Sb)=bin_x(S)

    Qs=Qs_of_S(r,Sb)
    display((Y, Qs))

    Printf.@printf(io, "%f %f\n", Y, Qs)

end

function JIMWLK_evolution(V,Y_f,ΔY)
    ξ_c = zeros(Float32, (2,N,N,Nc^2-1))
    ξ=zeros(ComplexF32, (2,N,N,Nc,Nc))
    ξ_k=zeros(ComplexF32, (2,N,N,Nc,Nc))
    ξ_conv_with_K=zeros(ComplexF32, (N,N,Nc,Nc))
    exp_R=zeros(ComplexF32, (N,N,Nc,Nc))
    exp_L=zeros(ComplexF32, (N,N,Nc,Nc))


    K_of_k=zeros(ComplexF32, (2,N,N))
    K_of_k_1 = @view K_of_k[1,:,:]
    K_of_k_2 = @view K_of_k[2,:,:]
    WW_kernel!(1,K_of_k_1)
    WW_kernel!(2,K_of_k_2)

    Y=0.0
    open("output_$ID.dat","w") do io
        while Y<Y_f
            observables(io, Y,V)

            generate_noise_Fourier_Space!(ξ,ξ_k,ξ_c)
            exp_Left(ξ_k, K_of_k, ξ_conv_with_K, exp_L, ΔY)

            rotated_noise(ξ,ξ_k,V)
            exp_Right(ξ_k, K_of_k, ξ_conv_with_K, exp_R, ΔY)

            Threads.@threads for i in 1:N
                for j in 1:N
                    @inbounds V_ij = @view V[i,j,:,:]
                    @inbounds exp_L_ij=@view exp_L[i,j,:,:]
                    @inbounds exp_R_ij=@view exp_R[i,j,:,:]
                    @fastmath V_ij .=   exp_L_ij*V_ij*exp_R_ij
                end
            end

            Y=Y+ΔY
        end
    end

end



V=compute_path_ordered_fund_Wilson_line()

Vc=compute_field_of_V_components(V)

JIMWLK_evolution(V,1.0,0.001)
