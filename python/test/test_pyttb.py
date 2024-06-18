try:
    import pyttb as ttb
    import numpy as np
    have_ttb = True
except:
    have_ttb = False

import pygenten as gt
import pytest

if have_ttb:

    def test_ttb_to_gt_tensor():

        # Make a dense pyttb tensor
        data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
        shape = (2,3,2)
        data = np.reshape(data, np.array(shape), order='F')
        x = ttb.tensor(data, shape)

        # Make a pygenten tensor from it as a copy, and verify it is a copy
        x_gt = gt.make_gt_tensor(x, copy=True)
        assert x_gt.is_values_view == False
        with np.nditer(x.data, flags=['multi_index']) as it:
            assert x.data[it.multi_index] == x_gt.data[it.multi_index]
        x_gt.data[0,0,0] = -1.0;
        assert x_gt.data[0,0,0] == -1.0
        assert x.data[0,0,0] != -1.0

        # Make a pygenten tensor from it as a view, and verify it is a view
        x_gt = gt.make_gt_tensor(x, copy=False)
        assert x_gt.is_values_view == True
        with np.nditer(x.data, flags=['multi_index']) as it:
            assert x.data[it.multi_index] == x_gt.data[it.multi_index]
        x_gt.data[0,0,0] = -2.0;
        assert x_gt.data[0,0,0] == -2.0
        assert x.data[0,0,0] == -2.0

    def test_gt_to_ttb_tensor():

        # Make a dense gt tensor, which must use 'F' layout
        data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
        shape = (2,3,2)
        data = np.reshape(data, np.array(shape), order='F')
        x_gt = gt.Tensor(data, copy=False)

        # Make a ttb tensor from it as a copy, and verify it is a copy
        x = gt.make_ttb_tensor(x_gt, copy=True)
        with np.nditer(x.data, flags=['multi_index']) as it:
            assert x.data[it.multi_index] == x_gt.data[it.multi_index]
        x.data[0,0,0] = -1.0;
        assert x.data[0,0,0] == -1.0
        assert x_gt.data[0,0,0] != -1.0

        # Make a ttb tensor from it as a view, and verify it is a view
        x = gt.make_ttb_tensor(x_gt, copy=False)
        with np.nditer(x.data, flags=['multi_index']) as it:
            assert x.data[it.multi_index] == x_gt.data[it.multi_index]
        x.data[0,0,0] = -2.0;
        assert x.data[0,0,0] == -2.0
        assert x_gt.data[0,0,0] == -2.0

    def test_ttb_to_gt_sptensor_unsigned():

        # Make a pyttb sptensor
        subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]], dtype='uint64')
        vals = np.array([[0.5], [1.5], [2.5], [3.5]])
        shape = (4, 4, 4)
        x = ttb.sptensor(subs, vals, shape)

        # Make a pygenten sptensor from it as a copy, and verify it is a copy
        x_gt = gt.make_gt_sptensor(x, copy=True)
        assert x_gt.is_values_view == False
        assert x_gt.is_subs_view == False
        nnz = x_gt.nnz
        nd = x_gt.ndims
        for i in range(nnz):
            assert x.vals[i] == x_gt.vals[i]
            for j in range(nd):
                assert x.subs[i,j] == x_gt.subs[i,j]
        x_gt.vals[0] = -1.0
        assert x_gt.vals[0] == -1.0
        assert x.vals[0] != -1.0
        x_gt.subs[0,0] = 3
        assert x_gt.subs[0,0] == 3
        assert x.subs[0,0] != 3

        # Make a pygenten sptensor from it as a view, and verify it is a view
        x_gt = gt.make_gt_sptensor(x, copy=False)
        assert x_gt.is_values_view == True
        assert x_gt.is_subs_view == True
        nnz = x_gt.nnz
        nd = x_gt.ndims
        for i in range(nnz):
            assert x.vals[i] == x_gt.vals[i]
            for j in range(nd):
                assert x.subs[i,j] == x_gt.subs[i,j]
        x_gt.vals[0] = -1.0
        assert x_gt.vals[0] == -1.0
        assert x.vals[0] == -1.0
        x_gt.subs[0,0] = 3
        assert x_gt.subs[0,0] == 3
        assert x.subs[0,0] == 3

    def test_ttb_to_gt_sptensor_signed():

        # Make a pyttb sptensor
        subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]], dtype='int64')
        vals = np.array([[0.5], [1.5], [2.5], [3.5]])
        shape = (4, 4, 4)
        x = ttb.sptensor(subs, vals, shape)

        # Make a pygenten sptensor from it as a copy, and verify it is a copy
        x_gt = gt.make_gt_sptensor(x, copy=True)
        assert x_gt.is_values_view == False
        assert x_gt.is_subs_view == False
        nnz = x_gt.nnz
        nd = x_gt.ndims
        for i in range(nnz):
            assert x.vals[i] == x_gt.vals[i]
            for j in range(nd):
                assert x.subs[i,j] == x_gt.subs[i,j]
        x_gt.vals[0] = -1.0
        assert x_gt.vals[0] == -1.0
        assert x.vals[0] != -1.0
        x_gt.subs[0,0] = 3
        assert x_gt.subs[0,0] == 3
        assert x.subs[0,0] != 3

        # Make a pygenten sptensor from it as a view, and vals is a view
        # but subs isn't since types do not match
        x_gt = gt.make_gt_sptensor(x, copy=False)
        assert x_gt.is_values_view == True
        assert x_gt.is_subs_view == False
        nnz = x_gt.nnz
        nd = x_gt.ndims
        for i in range(nnz):
            assert x.vals[i] == x_gt.vals[i]
            for j in range(nd):
                assert x.subs[i,j] == x_gt.subs[i,j]
        x_gt.vals[0] = -1.0
        assert x_gt.vals[0] == -1.0
        assert x.vals[0] == -1.0
        x_gt.subs[0,0] = 3
        assert x_gt.subs[0,0] == 3
        assert x.subs[0,0] != 3

    def test_gt_to_ttb_sptensor():

        # Make a pyttb sptensor
        subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]], dtype='uint64')
        vals = np.array([0.5, 1.5, 2.5, 3.5])
        shape = (4, 4, 4)
        x_gt = gt.Sptensor(shape, subs, vals)

        # Make a ttb sptensor from it as a copy, and verify it is a copy
        x = gt.make_ttb_sptensor(x_gt, copy=True)
        nnz = x_gt.nnz
        nd = x_gt.ndims
        for i in range(nnz):
            assert x.vals[i] == x_gt.vals[i]
            for j in range(nd):
                assert x.subs[i,j] == x_gt.subs[i,j]
        x.vals[0] = -1.0
        assert x.vals[0] == -1.0
        assert x_gt.vals[0] != -1.0
        x.subs[0,0] = 3
        assert x.subs[0,0] == 3
        assert x_gt.subs[0,0] != 3

        # Make a pygenten sptensor from it as a view, and verify it is a view
        x = gt.make_ttb_sptensor(x_gt, copy=False)
        nnz = x_gt.nnz
        nd = x_gt.ndims
        for i in range(nnz):
            assert x.vals[i] == x_gt.vals[i]
            for j in range(nd):
                assert x.subs[i,j] == x_gt.subs[i,j]
        x.vals[0] = -1.0
        assert x.vals[0] == -1.0
        assert x_gt.vals[0] == -1.0
        x.subs[0,0] = 3
        assert x.subs[0,0] == 3
        assert x_gt.subs[0,0] == 3

    def test_ttb_to_gt_ktensor():

        # Make a pyttb ktensor
        rank = 2
        weights = np.arange(1.0, rank+1)
        fm0 = np.array([[1., 3.], [2., 4.]])
        fm1 = np.array([[5., 8.], [6., 9.], [7., 10.]])
        fm2 = np.array([[11., 15.], [12., 16.], [13., 17.], [14., 18.]])
        factor_matrices = [fm0, fm1, fm2]
        u = ttb.ktensor(factor_matrices, weights)

        # Make a pygenten ktensor from it as a copy, and verify it is a copy
        u_gt = gt.make_gt_ktensor(u, copy=True)
        assert u_gt.is_values_view == False
        for j in range(rank):
            assert u.weights[j] == u_gt.weights[j]
        for k in range(len(factor_matrices)):
            for i in range(factor_matrices[k].shape[0]):
                for j in range(factor_matrices[k].shape[1]):
                    assert u.factor_matrices[k][i,j] == u_gt.factor_matrices[k][i,j]
        u_gt.weights[0] = -1.0;
        assert u_gt.weights[0] == -1.0
        assert u.weights[0] != -1.0
        u_gt.factor_matrices[0][0,0] = -1.0
        assert u_gt.factor_matrices[0][0,0] == -1.0
        assert u.factor_matrices[0][0,0] != -1.0

        # Make a pygenten ktensor from it as a view, and verify it is a view
        u_gt = gt.make_gt_ktensor(u, copy=False)
        assert u_gt.is_values_view == True
        for j in range(rank):
            assert u.weights[j] == u_gt.weights[j]
        for k in range(len(factor_matrices)):
            for i in range(factor_matrices[k].shape[0]):
                for j in range(factor_matrices[k].shape[1]):
                    assert u.factor_matrices[k][i,j] == u_gt.factor_matrices[k][i,j]
        u_gt.weights[0] = -1.0;
        assert u_gt.weights[0] == -1.0
        assert u.weights[0] == -1.0
        u_gt.factor_matrices[0][0,0] = -1.0
        assert u_gt.factor_matrices[0][0,0] == -1.0
        assert u.factor_matrices[0][0,0] == -1.0

    def test_gt_to_ttb_ktensor():

        # Make a pyttb ktensor
        rank = 2
        weights = np.arange(1.0, rank+1)
        fm0 = np.array([[1., 3.], [2., 4.]])
        fm1 = np.array([[5., 8.], [6., 9.], [7., 10.]])
        fm2 = np.array([[11., 15.], [12., 16.], [13., 17.], [14., 18.]])
        factor_matrices = [fm0, fm1, fm2]
        u_gt = gt.Ktensor(weights, factor_matrices)

        # Make a ttb ktensor from it as a copy, and verify it is a copy
        u = gt.make_ttb_ktensor(u_gt, copy=True)
        for j in range(rank):
            assert u.weights[j] == u_gt.weights[j]
        for k in range(len(factor_matrices)):
            for i in range(factor_matrices[k].shape[0]):
                for j in range(factor_matrices[k].shape[1]):
                    assert u.factor_matrices[k][i,j] == u_gt.factor_matrices[k][i,j]
        u.weights[0] = -1.0;
        assert u.weights[0] == -1.0
        assert u_gt.weights[0] != -1.0
        u.factor_matrices[0][0,0] = -1.0
        assert u.factor_matrices[0][0,0] == -1.0
        assert u_gt.factor_matrices[0][0,0] != -1.0

        # Make a ttb ktensor from it as a view, and verify it is a view
        u_gt = gt.make_ttb_ktensor(u, copy=False)
        for j in range(rank):
            assert u.weights[j] == u_gt.weights[j]
        for k in range(len(factor_matrices)):
            for i in range(factor_matrices[k].shape[0]):
                for j in range(factor_matrices[k].shape[1]):
                    assert u.factor_matrices[k][i,j] == u_gt.factor_matrices[k][i,j]
        u.weights[0] = -1.0;
        assert u.weights[0] == -1.0
        assert u_gt.weights[0] == -1.0
        u.factor_matrices[0][0,0] = -1.0
        assert u.factor_matrices[0][0,0] == -1.0
        assert u_gt.factor_matrices[0][0,0] == -1.0
