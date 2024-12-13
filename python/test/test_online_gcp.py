try:
    import pyttb as ttb
    have_ttb = True
except:
    have_ttb = False

if have_ttb:
    import pygenten as gt
    import numpy as np
    import copy as cpy
    import pytest

    def t_slice(u,t):
        # take a slice of a ktensor from a given t index
        ut = u.copy()
        tmp = ut.factor_matrices[-1]
        tmp = np.reshape(tmp[t,:],(1,-1))
        ut.factor_matrices[-1] = tmp
        return ut

    class SynGaussTensor:
        def __init__(self, shape, T, rank, noise):
            self.shape = shape
            self.T = T
            self.rank = rank
            self.noise = noise

            full_shape = self.shape + (T,)

            # construct underlying ktensor
            self.u_true = ttb.ktensor.from_function(np.random.random_sample, full_shape, rank)
            self.u_true.weights = np.random.random_sample(rank)
            self.u_true.normalize(None).redistribute(self.u_true.ndims-1) # normalize factor matrix columns and put all weight in time mode

            # construct full unperturbed tensor
            self.X = self.u_true.full()

            # time slice counter
            self.t = 0

        def draw(self):
            # reset counter based on period
            if self.t >= self.T:
                self.t = 0

            # get slice t of unperturbed data including a unit-length time-mode
            ind = [slice(None)]*self.X.ndims 
            ind[-1] = self.t
            Xt = self.X.data[tuple(ind)] # same as X[:,:,...,t]
            Xt = np.reshape(Xt,Xt.shape+(1,))
            Xt = ttb.tensor(Xt)

            # get slice t of corressponding ktensor
            ut = t_slice(self.u_true,self.t)

            # construct noise tensor for slice t
            Zt = ttb.tensor.from_function(np.random.standard_normal, self.shape+(1,))

            Xt += (self.noise*Xt.norm()/Zt.norm())*Zt

            # increment time counter
            self.t += 1

            return Xt,ut

    def test_online_gcp_dense_gaussian():
        syn_noise = 0.2        # noise level
        syn_rank = 2           # rank of underlying tensor
        syn_shape = (30,30)    # spatial size of the tensor
        syn_T = 20             # temporal period
        num_time = syn_T       # number of time steps in the final tensor
        num_warm = 10          # number of time steps used in the warm start
        rank = syn_rank        # rank of the CP decomposition

        # generate synthetic data
        np.random.seed(13) # set seed so we always get the same tensor
        syn = SynGaussTensor(syn_shape, syn_T, syn_rank, syn_noise)
        all_data = np.zeros((*syn_shape,syn_T))
        slices = []
        u_true_slice = []
        for t in range(num_time):
            data,ut = syn.draw()
            all_data[:,:,t] = data.data[:,:,0]
            slices.append(data)
            u_true_slice.append(ut)
        X = ttb.tensor(all_data)
        X0 = X[:,:,0:num_warm]

        # Generate warm start
        #u0 = syn.u_true.copy()
        #u0.factor_matrices[-1] = u0.factor_matrices[-1][0:num_warm,:]
        u_init,_,_ = gt.cp_als(X0,rank=rank,tol=1e-6,maxiters=1000)#,init=u0)
        u_init.normalize(None).redistribute(2) # normalize factor matrix columns and put all weight in time mode

        # Compute streaming GCP decomposition
        # temporal sgd solver params
        temporal_params = dict()
        temporal_params['sampling'] = 'uniform'
        temporal_params['fnzs'] = 100
        temporal_params['gnzs'] = 100
        temporal_params['hash'] = True
        temporal_params['fuse'] = True
        temporal_params['maxiters'] = 20
        temporal_params['fails'] = 3
        temporal_params['epochiters'] = 10
        temporal_params['rate'] = 10
        temporal_params['fit'] = True
        temporal_params['gcp-tol'] = 1e-5

        # spatial sgd solver params
        spatial_params = cpy.deepcopy(temporal_params)
        spatial_params['maxiters'] = 5
        spatial_params['rate'] = 1e-4

        # streaming solver params
        gcp_params = dict()
        gcp_params['streaming-solver'] = 'sgd'
        gcp_params['type'] = 'gaussian'
        gcp_params['rank'] = rank
        gcp_params['window-size'] = 10
        gcp_params['window-method'] = 'reservoir'
        gcp_params['printitn'] = 1

        # compute streaming gcp decomposition
        u_gcp,fest,ften = gt.online_gcp(slices[num_warm:],X0,u_init,temporal_params,spatial_params,**gcp_params)
        u_gcp.normalize(None).redistribute(2)
        u_gcp.factor_matrices[-1] = np.vstack((u_init.factor_matrices[-1],u_gcp.factor_matrices[-1]))

        # check similarity score with underlying k-tensor
        score = syn.u_true.score(u_gcp)[0]
        print(f'streaming gcp score = {score}')
        assert pytest.approx(0.9, abs=0.1) == score

        del u_init, u_gcp