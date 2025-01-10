#!/usr/bin/env python
# CURRENTLY WIP


import numpy as np


def minimum_image(cell, kpts):
    """
    Compute the minimum image of k-points in 'kpts' in the first Brillouin zone

    Arguments:
        cell -- a cell instance
        kpts -- a list of k-points

    Returns:
        kpts_bz -- a list of k-point in the first Brillouin zone
    """
    tmp_kpt = cell.get_scaled_kpts(kpts)
    tmp_kpt = tmp_kpt - np.floor(tmp_kpt)
    tmp_kpt[tmp_kpt > 0.5 - 1e-8] -= 1
    kpts_bz = cell.get_abs_kpts(tmp_kpt)
    return kpts_bz


def build_N_local_grid(N_local_x, N_local_y, N_local_z, Lvec_recip):
    if N_local_x % 2 == 1:
        Grid_1D_x = np.concatenate((np.arange(0, (N_local_x - 1) // 2 + 1), np.arange(-(N_local_x - 1) // 2, 0)))
    else:
        # At low Nlocal/Nk, this matters, because we want the direction where G is incremented to be opposite of
        # the default direction of a boundary-value q.
        Grid_1D_x = np.concatenate((np.arange(0, N_local_x // 2 + 1), np.arange(-N_local_x // 2 +1, 0)))
    if N_local_y % 2 == 1:
        Grid_1D_y = np.concatenate((np.arange(0, (N_local_y - 1) // 2 + 1), np.arange(-(N_local_y - 1) // 2, 0)))
    else:
        Grid_1D_y = np.concatenate((np.arange(0, N_local_y // 2 + 1), np.arange(-N_local_y // 2 +1, 0)))

    if N_local_z % 2 == 1:
        Grid_1D_z = np.concatenate((np.arange(0, (N_local_z - 1) // 2 + 1), np.arange(-(N_local_z - 1) // 2, 0)))
    else:
        Grid_1D_z = np.concatenate((np.arange(0, N_local_z // 2 + 1), np.arange(-N_local_z // 2 +1, 0)))

    Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D_x, Grid_1D_y, Grid_1D_z, indexing='ij')
    GptGrid3D_local = np.hstack(
        (Gxx_local.reshape(-1, 1), Gyy_local.reshape(-1, 1), Gzz_local.reshape(-1, 1))) @ Lvec_recip
    return GptGrid3D_local


# Define contracted gaussian model
def contracted_gaussian_model(params, xyz, num_gaussians=1):
    # Define Gaussian model function
    # xyz in shape (n, 3)
    num_gauss_params = 7 # per gaussian

    def gaussian_model(params, xyz):
        mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z = params
        exponent = -((xyz[:, 0] - mu_x) ** 2 / (2 * sigma_x ** 2) +
                     (xyz[:, 1] - mu_y) ** 2 / (2 * sigma_y ** 2) +
                     (xyz[:, 2] - mu_z) ** 2 / (2 * sigma_z ** 2))
        return np.exp(exponent) # if not subtract_nocc else -nocc + nocc * np.exp(exponent)

    assert len(params) == num_gauss_params * num_gaussians
    result = np.zeros(xyz.shape[0])
    for i in range(num_gaussians):
        c_i, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z = params[i * 7:(i + 1) * 7]
        result += c_i * gaussian_model([mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z], xyz)

    return result


def contracted_gaussian_model_centered(params, xyz, num_gaussians=1,isotropic=False,with_coul=False,method='scipy_minimize',c_0=0.0):
    # Define Gaussian model function
    # xyz in shape (n, 3)
    num_gauss_params = 4 # per gaussian
    if isotropic:
        num_gauss_params -= 2
    
    if with_coul:
        def gaussian_model(params, xyz):
            sigma_x, sigma_y, sigma_z = params
            exponent = -((xyz[:, 0]) ** 2 / (2 * sigma_x ** 2)
                         + (xyz[:, 1]) ** 2 / (2 * sigma_y ** 2)
                         + (xyz[:, 2]) ** 2 / (2 * sigma_z ** 2))
            return np.exp(exponent) / (np.sum(xyz**2,axis=1)) # if not subtract_nocc else -nocc + nocc * np.exp(exponent)
    else:
        def gaussian_model(params, xyz):
            sigma_x, sigma_y, sigma_z = params
            exponent = -((xyz[:, 0]) ** 2 / (2 * sigma_x ** 2) +
                         (xyz[:, 1]) ** 2 / (2 * sigma_y ** 2) +
                         (xyz[:, 2]) ** 2 / (2 * sigma_z ** 2))
            return np.exp(exponent)

    # assert len(params) == num_gauss_params * num_gaussians
    result = np.zeros(xyz.shape[0])

    if isotropic:
        for i in range(num_gaussians):
            c_i, sigma = params[i * num_gauss_params:(i + 1) * num_gauss_params]
            result += c_i * gaussian_model([sigma,sigma,sigma], xyz)
    else:
        for i in range(num_gaussians):
            c_i, sigma_x, sigma_y, sigma_z = params[i * num_gauss_params:(i + 1) * num_gauss_params]
            result += c_i * gaussian_model([sigma_x, sigma_y, sigma_z], xyz)

    return result

def fit_function_3d(xyz_input, f_input, nocc, subtract_nocc=False, num_gaussians=1, force_isotropic=False,
                    force_centered=False, with_coul=False, auto_guess=True, method="scipy_minimize"):

    # Initial guess for parameters
    # initial_guess = [nocc/num_gaussians, 0.0, 0.0, 0.0,
    #                  np.std(xyz_input[:, 0]), np.std(xyz_input[:, 1]), np.std(xyz_input[:, 2])] * num_gaussians

    if force_centered:
        if force_isotropic:
            initial_guess = [1./num_gaussians, 1.5] * num_gaussians
            if auto_guess:
                # Find index that has closest value to np.exp(-1./2) or 1 sigma away
                stds = 2.
                target = np.exp(-stds**2 / 2.0)

                # Find the index of the closest value
                target_index = np.argmin(np.abs(f_input - target))
                a0 = np.linalg.norm(xyz_input[target_index,:])
                a0 *= 1/stds
                if a0 < 0.1:
                    a0 = 1.25

            # a0 = 1.25
            beta = 0.5
            sigmas = [a0 * beta ** i for i in range(num_gaussians)]
            initial_guess[1::2] = sigmas
            num_gauss_params = 2
            offset = 0

            sigma_indices = [1]
        else:

            initial_guess = [1./num_gaussians, 1.5, 1.5, 1.5] * num_gaussians
            num_gauss_params = 4
            offset = 0

            sigma_indices = [1, 2, 3]

        def residuals(params_input, xyz, f, pow=2, method=method):
            if method == "scipy_least_squares":
                params = np.zeros(len(params_input)+1)
                params[1:] = params_input
                params[0] = 1 - np.sum(params_input[num_gauss_params::num_gauss_params])
            elif method == "scipy_minimize":
                params = params_input
            else:
                raise ValueError(f"Method {method} not recognized")

            f_fit = contracted_gaussian_model_centered(params, xyz, num_gaussians=num_gaussians,
                                                       isotropic=force_isotropic, with_coul=with_coul,method=method)
            return np.sum(np.abs(f_fit - f)**pow)

    else:
        initial_guess = [1./num_gaussians, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5] * num_gaussians
        num_gauss_params = 7
        offset = 3
        sigma_indices = [4, 5, 6]

        # Perform the curve fitting
        def residuals(params, xyz, f):
            # return contracted_gaussian_model(params, xyz, num_gaussians=num_gaussians) - f
            # Least squares
            return np.sum((contracted_gaussian_model(params, xyz, num_gaussians=num_gaussians) - f) ** 2)

    # Constraint where all c_i must be positive and sum to 1
    def normalization(params):
        return np.sum(params[::num_gauss_params]) - 1

    constraints = [
        {'type': 'eq', 'fun': normalization},
    ]

    # Force positive c and sigma values
    single_bound = [(-np.inf, np.inf)] * num_gauss_params
    single_bound[0] = (0.0,None)
    # for index in sigma_indices:
    #     single_bound[index] = (0.0, None)
    bounds = single_bound * num_gaussians

    from scipy.optimize import least_squares,minimize
    if method == "scipy_minimize":
        result = minimize(residuals, initial_guess, args=(xyz_input, f_input/nocc), constraints=constraints,bounds=bounds)
        params = result.x

    elif method == "scipy_least_squares":
        # Modify inital guess to take into account normalization constraints
        initial_guess = initial_guess[1:]
        bounds = bounds[1:]
        if len(bounds) == 1:
            bounds = bounds[0]
        result = least_squares(residuals, initial_guess, args=(xyz_input, f_input/nocc), bounds=bounds)
        params = np.zeros(len(result.x)+1)
        params[1:] = result.x
        params[0] = 1 - np.sum(result.x[num_gauss_params-1::num_gauss_params])
    # Renormalize the c_i values
    params[::num_gauss_params] *= nocc

    # Print parameters for each gaussian
    if force_centered:
        for i in range(num_gaussians):
            for j in sigma_indices:
                if params[i*num_gauss_params+j] < 0:
                    params[i*num_gauss_params+j] = -params[i*num_gauss_params+j]
                    print(f' Sigma {j} negative, converting to positive')

            if force_isotropic:
                print(f'Gaussian {i+1} parameters: c = {params[i*num_gauss_params]:.6f}, mu_x = 0.0,'
                      f'sigma = {params[i*num_gauss_params+1]:.6f}')
            else:
            
                print(f'Gaussian {i+1} parameters: c = {params[i*num_gauss_params]:.6f}, mu_x = 0.0, mu_y = 0.0, mu_z = 0.0,'
                    f'sigma_x = {params[i*num_gauss_params+1]:.6f}, sigma_y = {params[i*num_gauss_params+2]:.6f},'
                    f'sigma_z = {params[i*num_gauss_params+3]:.6f}')
            # If sigmas are negative, make them positive

    else:
        for i in range(num_gaussians):
            for j in sigma_indices:
                if params[i*num_gauss_params+j] < 0:
                    params[i*num_gauss_params+j] = -params[i*num_gauss_params+j]
                    print(f' Sigma {j} negative, converting to positive')

            print(f'Gaussian {i+1} parameters: c = {params[i*num_gauss_params]:.6f}, mu_x = {params[i*num_gauss_params+1]:.6f}, mu_y = {params[i*num_gauss_params+2]:.6f}, mu_z = {params[i*num_gauss_params+3]:.6f}, '
                 f'sigma_x = {params[i*num_gauss_params+4]:.6f}, sigma_y = {params[i*num_gauss_params+5]:.6f}, sigma_z = {params[i*num_gauss_params+6]:.6f}')

    return params

def compute_SqG_anisotropy(mf, cell, nks=np.array([3,3,3]), N_local=7, dim=3, dm_kpts=None, mo_coeff_kpts=None,
                           SqG_filename=None, num_gaussians=1, return_all_params=False,force_centered=True,
                           force_isotropic=False, fit_with_coul=False,auto_guess=True,fit_method="scipy_minimize",
                           debug=True,qG_norm_cutoff=None):
    # Perform a smaller calculation of the same system to get the anisotropy of SqG\
    print('Computing SqG anisotropy')

    if np.isscalar(N_local):
        N_local = np.array([N_local]*dim)
    else:
        N_local = np.array(N_local)
    N_local_x, N_local_y, N_local_z = N_local

    kpts = mf.kpts
    # kpts = cell.make_kpts(nks,wrap_around=True)
    nkpts = np.prod(nks)

    # Nk = np.prod(kmesh)
    if dm_kpts is None and SqG_filename is None:
        df_type = df.GDF
        mf.with_df = df_type(cell, kpts).build()
        e1 = mf.kernel()
        dm_kpts = mf.make_rdm1()
        if mo_coeff_kpts is None:
            mo_coeff_kpts = np.array(mf.mo_coeff_kpts)
    elif mo_coeff_kpts is None:
        raise ValueError("mo_coeff_kpts must be provided if dm_kpts is provided")

    # E_standard, E_madelung, uKpts, qGrid, kGrid = make_ss_inputs(kmf=mf, kpts=kpts, dm_kpts=dm_kpts,
    #                                                              mo_coeff_kpts=mo_coeff_kpts)

    uKpts = build_uKpts(mf, kpts, dm_kpts, mo_coeff_kpts)
    nocc = cell.tot_electrons() // 2

    #   Step 1.1: evaluate AO on a real fine mesh in unit cell
    Lvec_recip = cell.reciprocal_vectors()
    Lvec_real = mf.cell.lattice_vectors()
    NsCell = mf.cell.mesh
    L_delta = Lvec_real / NsCell[:, None]
    dvol = np.abs(np.linalg.det(L_delta))

    # Evaluate wavefunction on all real space grid points
    # # Establishing real space grid (Generalized for arbitary volume defined by 3 vectors)
    xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
    mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
    rptGrid3D = mesh_idx @ L_delta
    # aoval = kmf.cell.pbc_eval_gto("GTOval_sph", coords=rptGrid3D, kpts=kmf.kpts)

    #   Step 1.2: map q-mesh and k-mesh to BZ
    qGrid = minimum_image(cell, kpts - kpts[0, :])
    kGrid = minimum_image(cell, kpts)

    # assert that qGrid has origin
    if np.linalg.norm(qGrid[0]) > 1e-8:
        raise ValueError("Anisotropy calculation has support for qGrid with origin only")

    #   Step 1.3: evaluate MO periodic component on a real fine mesh in unit cell
    nbands = nocc
    nG = np.prod(NsCell)

    #   Step 1.4: compute the pair product
    Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
    Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
    Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
    Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
    GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip

    if SqG_filename is not None:
        if SqG_filename.endswith('.npy'):
            SqG = np.load(SqG_filename)
        elif SqG_filename.endswith('.pkl'):
            import pickle
            with open(SqG_filename, 'rb') as f:
                SqG = pickle.load(f)
        else:
            raise ValueError("SqG filename must end with .npy or pkl")
    else:
        SqG = build_SqG(nkpts, nG, nbands, kGrid, qGrid, mf, uKpts, rptGrid3D, dvol, NsCell, GptGrid3D)

    # Assert SqG at the origin is nocc
    assert np.abs(SqG[0, 0] - nocc) < 1e-4

    if debug:
        # section to manually compute exchange energy
        qG_full = np.zeros((nG * nkpts, 3))
        SqG_full = np.zeros(nG * nkpts)

        for iq in range(qGrid.shape[0]):
            qG = qGrid[iq, :] + GptGrid3D
            start_idx = iq * nG
            end_idx = (iq + 1) * nG
            qG_full[start_idx:end_idx, :] = qG
            SqG_full[start_idx:end_idx] = SqG[iq, :]

        qG2 = np.linalg.norm(qG_full,axis=1)**2
        summand = SqG_full/qG2
        summand[np.isinf(summand)] = 0

        b = cell.reciprocal_vectors()/nks
        weights = abs(np.linalg.det(b))
        weights *= 1/(2*np.pi)**3

        E_ex = - 4 * np.pi * np.sum(summand) * weights
        print(f'DEBUG: Exchange energy from SqG is {E_ex:.6f}')

    #   Reciprocal lattice within the local domain
    GptGrid3D_local = build_N_local_grid(N_local_x, N_local_y, N_local_z, Lvec_recip)

    #   location/index of GptGrid3D_local within 'GptGrid3D'
    idx_GptGrid3D_local = []
    for Gl in GptGrid3D_local:
        idx_tmp = np.where(np.linalg.norm(Gl[None, :] - GptGrid3D, axis=1) < 1e-8)[0]
        if len(idx_tmp) != 1:
            raise TypeError("Cannot locate local G vector in the reciprocal lattice.")
        else:
            idx_GptGrid3D_local.append(idx_tmp[0])
    idx_GptGrid3D_local = np.array(idx_GptGrid3D_local)

    #   focus on S(q + G) with q in qGrid and G in GptGrid3D_local
    SqG = SqG[:, idx_GptGrid3D_local]

    # Fit Gaussian to data
    nqG_local_3D = np.prod(N_local * nks)
    N_local_3D = np.prod(N_local)
    qG_full = np.zeros((nqG_local_3D, 3))
    SqG_local_full = np.zeros(nqG_local_3D)

    # Fill arrays with data
    for iq in range(qGrid.shape[0]):
        qG = qGrid[iq, :] + GptGrid3D_local
        start_idx = iq * N_local_3D
        end_idx = (iq + 1) * N_local_3D
        qG_full[start_idx:end_idx, :] = qG
        if fit_with_coul:
            SqG_local_full[start_idx:end_idx] = SqG[iq, :]/np.sum(qG**2,axis=1)
        else:
            SqG_local_full[start_idx:end_idx] = SqG[iq, :]

    # Remove inf and nan values
    qG_full = qG_full[~np.isinf(SqG_local_full)]
    SqG_local_full = SqG_local_full[~np.isinf(SqG_local_full)]

    # Restrict fitting to qG with norm less than qG_norm_cutoff
    if qG_norm_cutoff is not None:
        print("Using qG norm cutoff, fitting to qG with norm less than", qG_norm_cutoff)
        # If norm cutoff within the Nlocal BZs, print warning
        if max(np.linalg.norm(cell.reciprocal_vectors()*N_local,axis=1)) < qG_norm_cutoff:
            print("NOTE: qG_norm_cutoff is outside the longest dimension of the NlocalBZs")

        qG_norm = np.linalg.norm(qG_full,axis=1)
        SqG_local_full = SqG_local_full[qG_norm < qG_norm_cutoff]
        qG_full = qG_full[qG_norm < qG_norm_cutoff]

    # Fit Gaussian to data
    params = fit_function_3d(qG_full, SqG_local_full, nocc, subtract_nocc=False, num_gaussians=num_gaussians,
                             force_centered=force_centered, force_isotropic=force_isotropic,with_coul=fit_with_coul,
                             auto_guess=auto_guess,method=fit_method)

    # Extract sigma values or all parameters
    if return_all_params:
        return params
    else:
        sigmas = []
        for i in range(num_gaussians):
            sigmas.extend(params[i*7+3:i*7+6])
        return sigmas


def precompute_r1_prefactor(power_law_start,power_law_exponent,Nk,delta,gamma,M,r1,normal_vector):
    # # scaled_normal_vector = normal_vector*r1
    # qx = scaled_normal_vector[0]
    # qy = scaled_normal_vector[1]
    # qz = scaled_normal_vector[2]
    exp_term = (normal_vector*M)
    # r1 is the minimum distance from the center of the BZ to the boundary
    r1_prefactor_max = (-np.log(delta)/4)**(-1./2.) * np.sqrt(np.dot(exp_term,exp_term))
    r1_prefactor_min = (-np.log(gamma)/4)**(-1./2.) * np.sqrt(np.dot(exp_term,exp_term))

    def compute_r1_power_law(start,r1_max,r1_min,exponent,nk_1d):
        a = (r1_max-r1_min)*start**(-exponent)
        nk_1d = nk_1d.astype('float64')
        assert (exponent<0)
        return a*(nk_1d)**exponent + r1_min

    r1_prefactor_comp = compute_r1_power_law(power_law_start,r1_prefactor_max,r1_prefactor_min,power_law_exponent,Nk)
    # print(f'Precomputed r1 prefactor is {r1_prefactor_comp:.3f}')
    return r1_prefactor_comp


# make function determining distance from point to a plane containing the origin
def dist_to_plane(point, normal):
    return np.abs(np.dot(point, normal) / np.linalg.norm(normal))


def closest_fbz_distance(Lvec_recip,N_local):
    # Query point is the center of the parallelipid
    query_point = np.sum(Lvec_recip,axis=0) / 2
    # query_pooint = np.

    # For each unique pair of reciprocal vectors, find distance from the query point to the plane containing the origin
    distances = []
    pairs = []
    for i in range(3):
        for j in range(i+1, 3):
            distances.append(dist_to_plane(query_point, np.cross(Lvec_recip[i], Lvec_recip[j])))
            pairs.append((i,j))

    # Find the minimum distance
    r1 = np.min(N_local*distances) #must be scaled by nlocal
    return r1, pairs[np.argmin(distances)]


def build_SqG(nkpts, nG, nbands, kGrid, qGrid, kmf, uKpts, rptGrid3D, dvol, NsCell, GptGrid3D, nks=[1,1,1],
              subtract_nocc=0, debug_options={}):

    return build_SqG_k1k2(nkpts, nG, nbands, kGrid, kGrid, qGrid, kmf, uKpts, uKpts, rptGrid3D, dvol, NsCell,
                          GptGrid3D, nks=nks, subtract_nocc=subtract_nocc, debug_options=debug_options)


def build_SqG_k1k2(nkpts, nG, nbands, kGrid1,kGrid2, qGrid, kmf, uKpts1,uKpts2, rptGrid3D, dvol, NsCell,
                   GptGrid3D, nks=[1,1,1], subtract_nocc=0, debug_options={}):

    import os
    import numpy as np
    import scipy.io
    import time
    import pymp

    build_SqG_start_time = time.time()
    # SqG = pymp.shared.array((nkpts, nG), dtype=np.float64)
    SqG = np.zeros((nkpts, nG), dtype=np.float64)
    print("SqG MEM USAGE (KB) IS: {:.3f}".format( SqG.nbytes / (1024)))

    # nthreads = int(os.environ['OMP_NUM_THREADS'])
    # with pymp.Parallel(np.min([nthreads, 4])) as p:

    if debug_options:
        nqG = np.prod(NsCell)*nkpts
        qG_full = np.zeros([nqG,3])
        # HqG_local_full = np.zeros([nqG_local])
        SqG_full = np.zeros([nqG])
        # VqG_local_full = np.zeros([nqG_local])

    for q in range(nkpts):
        for k in range(nkpts):
            temp_SqG_k = np.zeros(nG, dtype=np.float64)  # Temporary storage for sums over m, n for the current k and q

            kpt1 = kGrid1[k, :]
            qpt = qGrid[q, :]
            kpt2 = kpt1 + qpt

            kpt2_BZ = minimum_image(kmf.cell, kpt2)
            idx_kpt2 = np.where(np.sum((kGrid2 - kpt2_BZ[None, :]) ** 2, axis=1) < 1e-8)[0]
            if len(idx_kpt2) != 1:
                raise TypeError("Cannot locate (k+q) in the kmesh.")
            idx_kpt2 = idx_kpt2[0]
            kGdiff = kpt2 - kpt2_BZ

            for n in range(nbands):
                for m in range(nbands):
                    u1 = uKpts1[k, n, :]
                    u2 = np.squeeze(np.exp(-1j * (rptGrid3D @ np.reshape(kGdiff, (-1, 1))))) * uKpts2[idx_kpt2, m, :]
                    rho12 = np.reshape(np.conj(u1) * u2, (NsCell[0], NsCell[1], NsCell[2]))
                    temp_fft = np.fft.fftn((rho12 * dvol))
                    temp_SqG_k += np.abs(temp_fft.reshape(-1)) ** 2

            SqG[q, :] += temp_SqG_k / nkpts

        if debug_options:
            qG = qpt[None, :] + GptGrid3D
            qG_full[q * nG:(q + 1) * nG] = qG
            SqG_full[q * nG:(q + 1) * nG] = SqG[q, :]

    build_SqG_end_time = time.time()
    print(f"Time to build SqG: {build_SqG_end_time - build_SqG_start_time} s")

    if debug_options:
        debug_options['filetype'] = debug_options.get('filetype', 'npy')
        debug_options['prefix'] = debug_options.get('prefix', '')

        if 'mat' in debug_options['filetype']:
            print('Saving qG mat files requested')
            scipy.io.savemat(debug_options['prefix'] + 'qG_full_nk' + str(nks[0]) + str(nks[1]) + str(nks[2]) + '.mat', {"qG_full": qG_full})
            scipy.io.savemat(debug_options['prefix'] + 'SqG_full_nk' + str(nks[0]) + str(nks[1]) + str(nks[2]) + '.mat', {"SqG_full": SqG_full})
        if 'pkl' in debug_options['filetype']:
            print('Saving qG pkl files requested')
            import pickle
            with open(debug_options['prefix']+'qGrid_nk' + str(nks[0]) + str(nks[1]) + str(nks[2]) + '.pkl', 'wb') as f:
                pickle.dump(qGrid, f)
            with open(debug_options['prefix']+'SqG_nk' + str(nks[0]) + str(nks[1]) + str(nks[2]) + '.pkl', 'wb') as f:
                pickle.dump(SqG, f)
        if 'npy' in debug_options['filetype']:
            print('Saving qG npy files requested')
            np.save(debug_options['prefix']+'qGrid_nk' + str(nks[0]) + str(nks[1]) + str(nks[2]) + '.npy', qGrid)
            np.save(debug_options['prefix']+'SqG_nk' + str(nks[0]) + str(nks[1]) + str(nks[2]) + '.npy', SqG)
        # raise ValueError('Debugging requested, halting calculation')

    return SqG


def build_uKpts(kmf, kpts, dm_kpts, mo_coeff_kpts):
    from pyscf.pbc.tools import get_monkhorst_pack_size

    # Setup constants
    NsCell = np.array(kmf.cell.mesh)
    nG = np.prod(NsCell)
    nocc = kmf.cell.tot_electrons() // 2
    nbands = nocc
    nk = get_monkhorst_pack_size(kmf.cell, kpts)
    Nk = np.prod(nk)

    # Setup real space grid points
    Lvec_real = kmf.cell.lattice_vectors()
    NsCell = np.array(kmf.cell.mesh)
    L_delta = Lvec_real / NsCell[:, None]
    dvol = np.abs(np.linalg.det(L_delta))
    xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
    mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
    rptGrid3D = mesh_idx @ L_delta

    # Evaluate the atomic orbitals at the real space grid points
    kGrid = minimum_image(kmf.cell, kpts)
    aoval = kmf.cell.pbc_eval_gto("GTOval_sph", coords=rptGrid3D, kpts=kpts)

    # Compute uKpts
    uKpts = np.zeros((Nk, nbands, nG), dtype=complex)
    for k in range(Nk):
        for n in range(nbands):
            utmp = aoval[k] @ np.reshape(mo_coeff_kpts[k][:, n], (-1, 1))
            exp_part = np.exp(-1j * (rptGrid3D @ np.reshape(kGrid[k], (-1, 1))))
            uKpts[k, n, :] = np.squeeze(exp_part * utmp)
    return uKpts

def make_ss_inputs(kmf,kpts,dm_kpts, mo_coeff_kpts):
    from pyscf.pbc.tools import madelung,get_monkhorst_pack_size
    import time
    ss_inputs_start = time.time()
    Madelung = madelung(kmf.cell, kpts)
    nocc = kmf.cell.tot_electrons() // 2
    nk = get_monkhorst_pack_size(kmf.cell, kpts)
    Nk = np.prod(nk)
    kmf.exxdiv = None
    _, K = kmf.get_jk(cell=kmf.cell, dm_kpts=dm_kpts, kpts=kpts, kpts_band=kpts,exxdiv=None)
    E_standard = -1. / Nk * np.einsum('kij,kji', dm_kpts, K) * 0.5
    E_standard /= 2
    E_madelung = E_standard - nocc * Madelung
    print(E_madelung)

    # Construct Grids 

    shiftFac=np.zeros(3)
    kshift_abs = np.sum(kmf.cell.reciprocal_vectors()*shiftFac / nk,axis=0)
    qGrid = minimum_image(kmf.cell, kshift_abs - kpts)
    kGrid = minimum_image(kmf.cell, kpts)

    # Construct the wavefunctions
    uKpts = build_uKpts(kmf, kpts, dm_kpts, mo_coeff_kpts)
    ss_inputs_end = time.time()
    print(f"Time taken for building uKpts: {ss_inputs_end - ss_inputs_start:.2f} s")
    return np.real(E_standard), np.real(E_madelung), uKpts, qGrid, kGrid


def madelung_modified(cell, kpts, shifted, ew_eta=None, anisotropic=False):
    # Here, the only difference from overleaf is that eta here is defined as 4eta^2 = eta_paper
    from pyscf.pbc.tools.pbc import get_monkhorst_pack_size
    from pyscf.pbc.gto.cell import get_Gv_weights

    printstr = "Modified Madelung correction"
    if anisotropic:
        printstr += " with anisotropy"
        assert not isinstance(ew_eta, int)
        raise NotImplementedError("Anisotropic Madelung correction not correctly implemented yet")

    print(printstr)
    # Make ew_eta into array to allow for anisotropy if len==3
    ew_eta = np.array(ew_eta)

    Nk = get_monkhorst_pack_size(cell, kpts)
    import copy
    cell_input = copy.copy(cell)
    cell_input._atm = np.array([[1, cell._env.size, 0, 0, 0, 0]])
    cell_input._env = np.append(cell._env, [0., 0., 0.])
    cell_input.unit = 'B' # ecell.verbose = 0
    cell_input.a = a = np.einsum('xi,x->xi', cell.lattice_vectors(), Nk)

    if ew_eta is None:
        ew_eta, _ = cell_input.get_ewald_params(cell_input.precision, cell_input.mesh)
    chargs = cell_input.atom_charges()
    log_precision = np.log(cell_input.precision / (chargs.sum() * 16 * np.pi ** 2))
    ke_cutoff = -2 * np.mean(ew_eta) ** 2 * log_precision
    ke_cutoff = min(ke_cutoff,cell.ke_cutoff) # Mesh shouldn't be bigger than the one used in the calculation
    # Get FFT mesh from cutoff value
    mesh = cell_input.cutoff_to_mesh(ke_cutoff)

    # Get grid
    Gv, Gvbase, weights = cell_input.get_Gv_weights(mesh=mesh)
    # Get q+G points
    G_combined = Gv + shifted
    absG2 = np.einsum('gi,gi->g', G_combined, G_combined)

    if cell_input.dimension ==3:
        # Calculate |q+G|^2 values of the shifted points
        qG2 = np.einsum('gi,gi->g', G_combined, G_combined)
        if anisotropic:
            denom = -1 / (4 * ew_eta ** 2)
            exponent = np.einsum('gi,gi,i->g', G_combined, G_combined, denom)
            exponent[exponent == 0] = -1e200
            component = 4 * np.pi / qG2 * np.exp(exponent)
        else:
            qG2[qG2 == 0] = 1e200
            component = 4 * np.pi / qG2 * np.exp(-qG2 / (4 * ew_eta ** 2))

        # First term
        sum_term = weights*np.einsum('i->',component).real
        # Second Term
        if anisotropic:
            from scipy.integrate import tplquad, nquad
            from cubature import cubature
            denom = -1 / (4 * ew_eta ** 2)

            # i denotes coordinate, g denotes vector number 
            def integrand(x, y, z):
                qG = np.array([x, y, z])
                denom = -1 / (4 * ew_eta ** 2)
                exponent = np.einsum('i,i,i->', qG, qG, denom)
                qG2 = np.einsum('i,i->', qG, qG)
                out = 4 * np.pi / qG2 * np.exp(exponent)

                # Handle special case when x, y, and z are very small
                if np.isscalar(out):
                    if (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12) & (np.abs(z) < 1e-12):
                        out = 0
                else:
                    mask = (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12) & (np.abs(z) < 1e-12)
                    out[mask] = 0  # gaussian case
                return out
            
            def integrand_vectorized(x,y,z):
                qG = np.array([x, y, z])
                denom = -1 / (4 * ew_eta ** 2)
                exponent = np.einsum('ig,ig,i->g', qG, qG, denom)
                qG2 = np.einsum('ig,ig->g', qG, qG)
                out = 4 * np.pi / qG2 * np.exp(exponent)

                # Handle special case when x, y, and z are very small
                if np.isscalar(out):
                    if (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12) & (np.abs(z) < 1e-12):
                        out = 0
                else:
                    mask = (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12) & (np.abs(z) < 1e-12)
                    out[mask] = 0  # gaussian case

                return out

        else:
            sub_term = 2*np.mean(ew_eta)/np.sqrt(np.pi)
        ewovrl = 0.0
        ewself_2 = 0.0
        print("Ewald components = %.15g, %.15g, %.15g,%.15g" % (ewovrl/2, sub_term/2,ewself_2/2, sum_term/2))
        return sub_term - sum_term

    elif cell_input.dimension == 2:  # Truncated Coulomb
        from scipy.special import erfc, erf

        # The following 2D ewald summation is taken from:
        # R. Sundararaman and T. Arias PRB 87, 2013
        def fn(eta, Gnorm, z):
            Gnorm_z = Gnorm * z
            large_idx = Gnorm_z > 20.0
            ret = np.zeros_like(Gnorm_z)
            x = Gnorm / 2. / eta + eta * z
            with np.errstate(over='ignore'):
                erfcx = erfc(x)
                ret[~large_idx] = np.exp(Gnorm_z[~large_idx]) * erfcx[~large_idx]
                ret[large_idx] = np.exp((Gnorm * z - x ** 2)[large_idx]) * erfcx[large_idx]
            return ret

        def gn(eta, Gnorm, z):
            return np.pi / Gnorm * (fn(eta, Gnorm, z) + fn(eta, Gnorm, -z))

        def gn0(eta, z):
            return -2 * np.pi * (z * erf(eta * z) + np.exp(-(eta * z) ** 2) / eta / np.sqrt(np.pi))

        b = cell_input.reciprocal_vectors()
        inv_area = np.linalg.norm(np.cross(b[0], b[1])) / (2 * np.pi) ** 2
        # Perform the reciprocal space summation over  all reciprocal vectors
        # within the x,y plane.
        planarG2_idx = np.logical_and(Gv[:, 2] == 0, absG2 > 0.0)

        G_combined = G_combined[planarG2_idx]
        absG2 = absG2[planarG2_idx]
        absG = absG2 ** (0.5)
        # Performing the G != 0 summation.
        coords = np.array([[0,0,0]])
        rij = coords[:, None, :] - coords[None, :, :] # should be just the zero vector for correction.
        Gdotr = np.einsum('ijx,gx->ijg', rij, G_combined)
        ewg = np.einsum('i,j,ijg,ijg->', chargs, chargs, np.cos(Gdotr),
                        gn(ew_eta, absG, rij[:, :, 2:3]))
        # Performing the G == 0 summation.
        # ewg += np.einsum('i,j,ij->', chargs, chargs, gn0(ew_eta, rij[:, :, 2]))

        ewg *= inv_area # * 0.5

        ewg_analytical = 2 * ew_eta / np.sqrt(np.pi)
        
        return ewg - ewg_analytical


def khf_ssng(mf, nks, num_gaussians=1, force_centered=True, force_isotropic=True, fit_with_coul=False, N_local=None, 
             sigma_multiplier=1.0,sigma=None,auto_guess=True,fit_method="scipy_minimize",qG_norm_cutoff=None):
    from pyscf.pbc.tools import madelung
    import time
    print("SS-NG Correction for Exact Exchange with sigma_multiplier = %.2f" % sigma_multiplier)
    dm_kpts = mf.make_rdm1()
    nocc = mf.cell.nelectron // 2
    kpts = mf.kpts
    nk = np.prod(nks)
    if N_local is None:
        N_local = mf.cell.mesh

    if force_isotropic:
        num_gaussian_params = 2
    else:
        num_gaussian_params = 4

    fit_start = time.time()
    if sigma is None:
        # Fit Gaussian to Structure Factor
        print('Fitting gaussian parameters... ')
        params = compute_SqG_anisotropy(mf,cell=mf.cell, nks=nks, N_local=N_local, dm_kpts=dm_kpts,
                                        mo_coeff_kpts=mf.mo_coeff_kpts, num_gaussians=num_gaussians,
                                        return_all_params=True, force_centered=force_centered,
                                        force_isotropic=force_isotropic, fit_with_coul=fit_with_coul,
                                        auto_guess=auto_guess,fit_method=fit_method,qG_norm_cutoff=qG_norm_cutoff)
        fit_end = time.time()
        params[1::num_gaussian_params] *= sigma_multiplier

        print('Fitting done in %.2f seconds' % (fit_end - fit_start))
        print('Fitting parameters: ', params)
    else:
        print('Using provided sigma values: ', sigma)
        params = np.zeros(num_gaussian_params * num_gaussians)
        params[::num_gaussian_params] = nocc/num_gaussians
        params[1::num_gaussian_params] = sigma

    # Compute Exchange Energies
    mf.exxdiv = None  # so that standard energy is computed without madelung
    J, K = mf.get_jk(cell=mf.cell, dm_kpts=dm_kpts, kpts=kpts, kpts_band=kpts, with_j=False, exxdiv=None)
    mf.exxdiv = 'ewald'

    Ek_uncorr = -1. / nk * np.einsum('kij,kji', dm_kpts, K) * 0.5
    Ek_uncorr /= 2.
    Ek_uncorr = Ek_uncorr.real

    chi_regular = madelung(mf.cell, kpts)
    Ek_regular = Ek_uncorr - nocc * chi_regular

    # Compute SS-NG exchange energy
    if num_gaussians == 1:
        shift = np.array([0., 0., 0.])
        assert len(params) == 2, "Two parameters required for modified madelung"
        c_i, sigma = params
        assert c_i == nocc, "c_i must be equal to nocc"
        ew_eta = 1. / np.sqrt(2.) * sigma

        chi = madelung_modified(mf.cell, kpts, shift, ew_eta=ew_eta, anisotropic=False)
        Ek = Ek_uncorr - nocc * chi
    else:
        shifted = np.array([0,0,0])
        num_gauss_params = 2
        chi = 0
        for i in range(num_gaussians):
            if num_gauss_params == 4:
                c_i, sigma_x, sigma_y, sigma_z = params[i*num_gauss_params:(i+1)*num_gauss_params]
            elif num_gauss_params == 2:
                c_i, sigma = params[i*num_gauss_params:(i+1)*num_gauss_params]
                sigma_x = sigma_y = sigma_z = sigma

            # Detect anisotropy
            anisotropic = False
            if np.abs(sigma_x - sigma_y) < 1e-8 and np.abs(sigma_y - sigma_z) < 1e-8:
                ew_eta_i = 1./np.sqrt(2.) * np.mean([sigma_x, sigma_y, sigma_z])# TODO: Implement anisotropy
            else:
                ew_eta_i = 1./np.sqrt(2.) * np.array([sigma_x, sigma_y, sigma_z])
                anisotropic = True

            chi_i = madelung_modified(mf.cell, kpts, shifted, ew_eta=ew_eta_i,anisotropic=anisotropic)
            chi = chi + c_i * chi_i
            print("Term ", i)
            if anisotropic:
                print(f" Input  sigma x = {sigma_x:.12f}, sigma y = {sigma_y:.12f}, sigma z = {sigma_z:.12f}")
            else:
                print(f" Input mean sigma: {np.mean([sigma_x, sigma_y, sigma_z]):.12f}")
            print(f" Input ew_eta:     {ew_eta_i:.12f}")
            print(f" Coefficient:      {c_i:.12f}")
            print(f" Chi:              {chi_i:.12f}")
            print(f" Contribution:     {c_i * chi_i:.12f}")
        Ek = Ek_uncorr - chi

    results = {
        'Ek_uncorr': Ek_uncorr,
        'Ek_probe': Ek_regular,
        'Ek_ss_ng': Ek,
    }

    print('khf_ss_ng results:')
    print(' Ek_uncorr = %.15g' % Ek_uncorr)
    print(' Ek_probe = %.15g' % Ek_regular)
    print(' Ek_ss_ng = %.15g' % Ek)

    print('Total time for SS-NG: %.2f seconds' % (time.time() - fit_start))
    return results