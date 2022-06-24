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
const N = 32
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

function generate_rho_fft_to_momentum_space(rho,fft_plan)
    rho .= variance_of_mv_noise * randn(rng, Float32,(N,N))
    rho .= fft_plan*rho
end

function inv_propagator_kernel(i,j)
	return (a2 / (a2 * m² + 4.0 * sin(π*i/N)^2 + 4.0 * sin(π*j/N)^2))
end

function inv_propagator(D)
    x = collect(0:N-1)
    y = copy(x)
    D .= map(Base.splat(inv_propagator_kernel), Iterators.product(x, y))
end 

function compute_field!(rhok, D, ifft_plan)
	# Modifies the argument to return the field
    rhok .= rhok .* D
    # factor of a^2 was removed to account for the normalization of ifft next
    # ifft computes sum / (lenfth of array) for each dimension
    rhok[1,1] = 0.0im # remove zero mode 	
    rhok .= ifft_plan*rhok
end

function compute_local_fund_Wilson_line()
    A_arr = zeros(ComplexF32, (N,N, Nc, Nc))
    V = zeros(ComplexF32, (N,N,Nc,Nc))

    ρ_k = zeros(ComplexF32,(N,N))
    fft_plan = plan_fft(ρ_k; flags=FFTW.MEASURE, timelimit=Inf)
    ifft_plan = plan_ifft(ρ_k; flags=FFTW.MEASURE, timelimit=Inf)
	D = zeros(Float32,(N,N))
	inv_propagator(D)

    for b in 1:Nc^2-1

        generate_rho_fft_to_momentum_space(ρ_k, fft_plan)
        compute_field!(ρ_k, D, ifft_plan)
        A = real.(ρ_k)

        Threads.@threads for j in 1:N
            for i in 1:N
				@inbounds A_arr_ij = @view A_arr[i,j,:,:]
                @inbounds A_arr_ij .= A_arr_ij + A[i,j]*t[b]
            end
        end
    end

    Threads.@threads for j in 1:N
        for i in 1:N
            @inbounds V_ij = @view V[i,j,:,:]
			@inbounds A_arr_ij = @view A_arr[i,j,:,:]
            @inbounds V_ij .= exp(1.0im.*A_arr_ij)
        end
    end
    return V
end


function compute_path_ordered_fund_Wilson_line()
    V = compute_local_fund_Wilson_line()
    for i in 1:Ny-1
		print(i) 
        tmp=compute_local_fund_Wilson_line()
        Threads.@threads for j in 1:N
            for i in 1:N
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
        for j in 1:N
            for i in 1:N
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

function k2_symm(i,j)
    return((sin(2π*(i-1)/N)^2 +  sin(2π*(j-1)/N)^2)/a^2)
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

function alpha_f(kx,ky)
    k2_s = k2_symm(kx,ky)
    #k2_s = k2(kx,ky)

    mu_over_LambdaQCD = 15.0/6.0
    LambdaQCD = 6.0/l

    beta = 9.0
    c=0.2

    return 4.0π/(beta*c*log(mu_over_LambdaQCD^(2.0/c)  + (k2_s/LambdaQCD^2)^(1.0/c) ))
end

function alphaArray()
    kx = collect(1:N)
    ky = copy(kx)
    α_k = map(Base.splat(alpha_f), Iterators.product(kx, ky))

	# to make this symmetric and all real 
	#α = real.(ifft(α_k))
	#return( real.(fft(α)) )
	return( α_k )
end

# It is more convenient to have \xi (z,x) where x is the last spatial variable
# ξ[polarization, z_1 ,z_2, x_1, x_2, color_a, color_b]

function generate_noise_Fourier_Space_mem_ef!(ξ, ξ_k,  α, fft_plan, ifft_plan)
    for ic in 1:Nc
        for jc in ic+1:Nc
            @inbounds ξ_icjc = @view ξ[:,:,:,:,:,ic,jc]
            @inbounds ξ_jcic = @view ξ[:,:,:,:,:,jc,ic]
            randn!(rng,ξ_icjc)         # generate noise
            ξ_icjc .= ξ_icjc/sqrt(2.0) # norm
            ξ_jcic .= conj.(ξ_icjc)    # hermitian
        end
    end

    ξ_1c1c = @view ξ[:,:,:,:,:,1,1]
    randn!(rng,ξ_1c1c)
    ξ_1c1c .= real.(ξ_1c1c)/sqrt(2.0)

    ξ_3c3c = @view ξ[:,:,:,:,:,3,3]
    randn!(rng,ξ_3c3c)
    ξ_3c3c .= -sqrt(2.0/3.0)*real.(ξ_3c3c)

    ξ_2c2c = @view ξ[:,:,:,:,:,2,2]
    ξ_2c2c .= -(ξ_1c1c .+ ξ_3c3c/2.0)

    ξ_1c1c .= (ξ_1c1c .- ξ_3c3c/2.0)

    ξ_k .= fft_plan*ξ
    #ξ_k .= ξ_k  # a^2/a ; 1/a *a is from the noise noise correlator


    @Threads.threads for ky in 1:N
        for kx in 1:N
			# working with x_1 and x_2 
            ξ_k_kx_ky = @view ξ_k[:,:,:,kx,ky,:,:]
            ξ_k_kx_ky .= 2π*sqrt(α[kx,ky])*ξ_k_kx_ky
        end
    end
	ξ .= (ifft_plan*ξ_k)/a^2
	#ξ_k .= (fft_plan*ξ)*a^2
	#
	\xi_k IS NOT FFTed in z!!!!
	
end

function convolution!(K, ξ_k, ξ_out, tmp, ifft_plan)
    @inbounds sub_ξ_1 = @view ξ_k[1,:,:,:,:,:,:]
    @inbounds sub_K_1 = @view K[1,:,:]
    @inbounds sub_ξ_2 = @view ξ_k[2,:,:,:,:,:,:]
    @inbounds sub_K_2 = @view K[2,:,:]
	# broadcasting over z  
    @fastmath tmp .= sub_ξ_1 .* sub_K_1 .+ sub_ξ_2 .* sub_K_2
	tmp .= (ifft_plan*tmp)/a2 
	for j in 1:N # cycle over x-s
        for i in 1:N
			ξ_out_ij = @view ξ_out[i,j,:,:] 
			tmp_ijij = @view tmp[i,j,i,j,:,:]
			ξ_out_ij .= tmp_ijij
        end
    end
end

function rotated_noise(ξ,ξ_R_k,V,fft_plan)
    Threads.@threads for j in 1:N
        for i in 1:N
           	@inbounds V_ij = @view V[i,j,:,:]
            aV = adjoint(V_ij)
			for ip = 1:N 
				for jp in 1:N
            		for p in 1:2
                		@inbounds ξ_pij = @view ξ[p,i,j,ip,jp,:,:]
                		@fastmath @inbounds ξ_pij .= aV*ξ_pij*V_ij
					end 
				end 
            end
        end
    end
	# fft in z -> k_z; plan in 2 and 3 coordinate
	ξ_R_k .= a2.*(fft_plan*ξ)
end

function exp_Left(ξ_k, K, ξ_out, exp_out, tmp,  ΔY, ifft_plan, prefactor)
    convolution!(K, ξ_k, ξ_out, tmp, ifft_plan)
    Threads.@threads for j in 1:N
        for i in 1:N
            @inbounds Exp = @view exp_out[i,j,:,:]
            @inbounds ξ_ij = @view ξ_out[i,j,:,:]
            @fastmath Exp .= exp(prefactor*1.0im*ξ_ij)
        end
    end
	println("xi_k")
	println( ξ_k[1,1,1,1,1,:,:] - adjoint( ξ_k[1,1,1,1,1,:,:] ))
	println("xi_out")
	println( ξ_out[1,1,:,:] - adjoint( ξ_out[1,1,:,:] ))
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

    println((Y, Qs))
    
	println("\n")
	
	println(S[1,1])

    Printf.@printf(io, "%f %f \n", Y, Qs)

end

function JIMWLK_evolution(V,Y_f,ΔY)

	sqrt_alpha_dY_over_pi=sqrt(alpha_fc*ΔY)/π
 
    ξ=zeros(ComplexF32, (2,N,N,N,N,Nc,Nc))
    ξ_k=zeros(ComplexF32, (2,N,N,N,N,Nc,Nc))
	tmp = zeros(ComplexF32, (N,N,N,N,Nc,Nc))
    ξ_conv_with_K=zeros(ComplexF32, (N,N,Nc,Nc))
    exp_R=zeros(ComplexF32, (N,N,Nc,Nc))
    exp_L=zeros(ComplexF32, (N,N,Nc,Nc))

    fft_plan = plan_fft(ξ,(4,5); flags=FFTW.MEASURE, timelimit=Inf)
    rotated_fft_plan = plan_fft(ξ,(2,3); flags=FFTW.MEASURE, timelimit=Inf)

    inv_fft_plan = plan_ifft(ξ_k,(4,5); flags=FFTW.MEASURE, timelimit=Inf)
    ifft_plan = plan_ifft(tmp,(1,2); flags=FFTW.MEASURE, timelimit=Inf)

    K_of_k=zeros(ComplexF32, (2,N,N))
    K_of_k_1 = @view K_of_k[1,:,:]
    K_of_k_2 = @view K_of_k[2,:,:]
    WW_kernel!(1,K_of_k_1)
    WW_kernel!(2,K_of_k_2)

	α=alphaArray()

    Y=0.0
    open("output_$ID.dat","w") do io
        while Y<Y_f
            observables(io, Y,V)

            #generate_noise_Fourier_Space!(ξ,ξ_k,ξ_c)
			println("Noise")
            generate_noise_Fourier_Space_mem_ef!(ξ,ξ_k,  α, fft_plan, inv_fft_plan)
            
			println("xi_k noise gen")
			println(ξ_k[1,1,1,1,1,:,:] - adjoint(ξ_k[1,1,1,1,1,:,:]))
			println("xi noise gen")
			println(ξ[1,1,1,1,1,:,:] - adjoint(ξ[1,1,1,1,1,:,:]))
			
			exp_Left(ξ_k, K_of_k, ξ_conv_with_K, exp_L, tmp, ΔY, ifft_plan, -sqrt_alpha_dY_over_pi)

			println("Rotated_Noise")
            rotated_noise(ξ,ξ_k,V,rotated_fft_plan) # rewrites original \xi
            # - sign to make it right
			println("xi_k Rot noise gen")
			println(ξ_k[1,1,1,1,1,:,:] - adjoint(ξ_k[1,1,1,1,1,:,:]))
			#
			exp_Left(ξ_k, K_of_k, ξ_conv_with_K, exp_R, tmp, ΔY, ifft_plan, sqrt_alpha_dY_over_pi)

            Threads.@threads for j in 1:N
                for i in 1:N
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



@time V=compute_path_ordered_fund_Wilson_line()


@time JIMWLK_evolution(V,1.0,0.001)
