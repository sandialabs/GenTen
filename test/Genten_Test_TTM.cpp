    // #include <gtest/gtest.h> 
#include <iostream>
#include <sstream>
#include "Genten_IOtext.hpp"
#include "Genten_TTM.hpp"
#include <string>
#include "Genten_Test_Utils.hpp"
#include "Genten_Util.hpp"


using namespace Genten::Test;

template <typename Space>
bool unit_test_tensor(Genten::TensorT<Space> Z, ttb_real *unit_test, int prod)
{
    for (int i = 0; i < prod; ++i)
    {
        // std::cout << Z.getValues().values()(i) <<std::endl;
        // EXPECT_NEAR(Z.getValues().values()(i), unit_test[i], 1e-10);
        if(Z.getValues().values()(i) != unit_test[i]){
            std::stringstream dim_error;
            dim_error << "Z[" << i << "]: " << Z.getValues().values()(i) << "   unit_test: " << unit_test[i];
            std::cerr << dim_error.str() << std::endl;
            throw dim_error.str();
            return false;
        }
        Z.getValues().values()(i) = 0; //Zeroing Z out so that we know each ttm implementation is changing Z
    }
    
    return true;
}

template<typename Space> 
int bulk_test(Genten::TensorT<Space> X, Genten::TensorT<Space> mat, int mode, ttb_real *unit_test)
{

    Genten::IndxArrayT<Space> result_size(X.ndims());
    deep_copy(result_size, X.size());
    result_size[mode] = mat.size(0);

    Genten::TensorT<Space> Z(result_size, 0.0);
    int prod = Z.size().prod(); 

    Genten::AlgParams al;

    std::cout << std::endl
              << "Testing default DGEMM cuBlas enabled ttm along mode: " << mode << std::endl;
#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
    Genten::TensorT<Genten::DefaultExecutionSpace> X_device = create_mirror_view(Genten::DefaultExecutionSpace(), X);
	deep_copy(X_device, X);
    Genten::TensorT<Genten::DefaultExecutionSpace> mat_device = create_mirror_view(Genten::DefaultExecutionSpace(), mat);
	deep_copy(mat_device, mat);
    Genten::TensorT<Genten::DefaultExecutionSpace> Z_device = create_mirror_view(Genten::DefaultExecutionSpace(), Z);
	deep_copy(Z_device, Z);

    Genten::ttm(X_device, mat_device, mode, Z_device, al);
    //Unload data off of device and check correctness	
	deep_copy(Z,Z_device);
    ASSERT(unit_test_tensor(Z, unit_test, prod), "test"); //NOTE: we need to copy data from device
    std::cout << std::endl
              << "Testing Parfor_DGEMM cuBlas enabled ttm along mode: " << mode << std::endl;
    al.ttm_method = Genten::TTM_Method::Parfor_DGEMM;
    Genten::ttm(X_device, mat_device, mode, Z_device, al);
    //Unload data off of device and check correctness	
	deep_copy(Z,Z_device);
    ASSERT(unit_test_tensor(Z, unit_test, prod), "test"); //NOTE: we need to copy data from device
#else
    Genten::ttm(X, mat, mode, Z, al);
    ASSERT(unit_test_tensor(Z, unit_test, prod), "test");
    std::cout << std::endl
              << "Testing Parfor_DGEMM cuBlas enabled ttm along mode: " << mode << std::endl;
    al.ttm_method = Genten::TTM_Method::Parfor_DGEMM;
    Genten::ttm(X, mat, mode, Z, al);
    ASSERT(unit_test_tensor(Z, unit_test, prod), "test");
    std::cout << std::endl
              << "Testing serial dgemm along mode: " << mode << std::endl;
    Genten::Impl::genten_ttm_serial_dgemm(mode, X, mat, Z);
    ASSERT(unit_test_tensor(Z, unit_test, prod), "test");
    std::cout << std::endl
              << "Testing parfor dgemm along mode: " << mode << std::endl;
    //Z.getValues().values()(0)=483294;
    Genten::Impl::genten_ttm_parfor_dgemm(mode, X, mat, Z);
    // Z.getValues().values()(0)=483294; //Sanity check
    ASSERT(unit_test_tensor(Z, unit_test, prod), "test");
#endif
    
    
    
    return 0;
}

// TEST(Test, ttm_test_0)
void test0()
{
    typedef Genten::DefaultHostExecutionSpace Space;
    typedef Genten::TensorT<Space> Tensor_type;
    typedef Genten::FacMatrixT<Space> FacMatrix_type;

    Genten::IndxArray tensor_dims;
    tensor_dims = Genten::IndxArray(4);
    tensor_dims[0] = 3;
    tensor_dims[1] = 4;
    tensor_dims[2] = 2;
    tensor_dims[3] = 2;

    Genten::IndxArray matrix_dims;
    matrix_dims = Genten::IndxArray(2);
    matrix_dims[0] = 5;
    matrix_dims[1] = 3;

    Tensor_type X(tensor_dims, 0.0);

    for (ttb_real i = 0; i < X.size().prod(); ++i)
    {
        X[i] = i;
    }

    Tensor_type mat(matrix_dims, 0.0);

    for (ttb_real i = 0; i < mat.size().prod(); ++i)
    {
        mat[i] = i;
    }
    ttb_indx mode = 0;

    Genten::IndxArrayT<Space> result_size(X.ndims());
    deep_copy(result_size, X.size());
    result_size[mode] = mat.size(0);

    Tensor_type Z(result_size, 0.0);
    // std::cout << "preparing to calculate ttm:  5x2 * 3x4x2x2 along mode 0" << std::endl;
    MESSAGE("preparing to calculate ttm:  5x2 * 3x4x2x2 along mode 0");
    
    int prod = Z.size().prod();

    ttb_real unit_test[120] = {25, 28, 31, 34, 37, 70, 82, 94, 106, 118, 115, 136, 157, 178, 199, 160, 190, 220, 250, 280, 205, 244, 283, 322, 361, 250, 298, 346, 394, 442, 295, 352, 409, 466, 523, 340, 406, 472, 538, 604, 385, 460, 535, 610, 685, 430, 514, 598, 682, 766, 475, 568, 661, 754, 847, 520, 622, 724, 826, 928, 565, 676, 787, 898, 1009, 610, 730, 850, 970, 1090, 655, 784, 913, 1042, 1171, 700, 838, 976, 1114, 1252};
    
    bulk_test( X,  mat, mode, unit_test);
}

// TEST(Test, ttm_test_1)
void test1()
{
    typedef Genten::DefaultHostExecutionSpace Space;
    typedef Genten::TensorT<Space> Tensor_type;
    typedef Genten::FacMatrixT<Space> FacMatrix_type;

    Genten::IndxArray tensor_dims;
    tensor_dims = Genten::IndxArray(4);
    tensor_dims[0] = 3;
    tensor_dims[1] = 4;
    tensor_dims[2] = 2;
    tensor_dims[3] = 2;

    Genten::IndxArray matrix_dims;
    matrix_dims = Genten::IndxArray(2);
    matrix_dims[0] = 5;
    matrix_dims[1] = 4;

    Tensor_type X(tensor_dims, 0.0);

    for (ttb_real i = 0; i < X.size().prod(); ++i)
    {
        X[i] = i;
    }

    Tensor_type mat(matrix_dims, 0.0);

    for (ttb_real i = 0; i < mat.size().prod(); ++i)
    {
        mat[i] = i;
    }
    ttb_indx mode = 1;

    Genten::IndxArrayT<Space> result_size(X.ndims());
    deep_copy(result_size, X.size());
    result_size[mode] = mat.size(0);

    Tensor_type Z(result_size, 0.0);
    std::cout << "preparing to calculate ttm:  5x4 * 3x4x2x2 along mode 1" << std::endl;

    int prod = Z.size().prod();

    ttb_real unit_test[60] = {210, 240, 270, 228, 262, 296, 246, 284, 322, 264, 306, 348, 282, 328, 374, 570, 600, 630, 636, 670, 704, 702, 740, 778, 768, 810, 852, 834, 880, 926, 930, 960, 990, 1044, 1078, 1112, 1158, 1196, 1234, 1272, 1314, 1356, 1386, 1432, 1478, 1290, 1320, 1350, 1452, 1486, 1520, 1614, 1652, 1690, 1776, 1818, 1860, 1938, 1984, 2030};


    bulk_test( X,  mat, mode, unit_test);
}

// TEST(Test, ttm_test_2)
void test2()
{
    typedef Genten::DefaultHostExecutionSpace Space;
    typedef Genten::TensorT<Space> Tensor_type;
    typedef Genten::FacMatrixT<Space> FacMatrix_type;

    Genten::IndxArray tensor_dims;
    tensor_dims = Genten::IndxArray(4);
    tensor_dims[0] = 3;
    tensor_dims[1] = 4;
    tensor_dims[2] = 2;
    tensor_dims[3] = 2;

    Genten::IndxArray matrix_dims;
    matrix_dims = Genten::IndxArray(2);
    matrix_dims[0] = 5;
    matrix_dims[1] = 2;

    Tensor_type X(tensor_dims, 0.0);

    for (ttb_real i = 0; i < X.size().prod(); ++i)
    {
        X[i] = i;
    }

    Tensor_type mat(matrix_dims, 0.0);

    for (ttb_real i = 0; i < mat.size().prod(); ++i)
    {
        mat[i] = i;
    }
    ttb_indx mode = 2;

    Genten::IndxArrayT<Space> result_size(X.ndims());
    deep_copy(result_size, X.size());
    result_size[mode] = mat.size(0);

    Tensor_type Z(result_size, 0.0);
    std::cout << "preparing to calculate ttm: 5x2 * 3x4x2x2 along mode 2" << std::endl;

    int prod = Z.size().prod();

    ttb_real unit_test[120] = {60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 72, 79, 86, 93, 100, 107, 114, 121, 128, 135, 142, 149, 84, 93, 102, 111, 120, 129, 138, 147, 156, 165, 174, 183, 96, 107, 118, 129, 140, 151, 162, 173, 184, 195, 206, 217, 108, 121, 134, 147, 160, 173, 186, 199, 212, 225, 238, 251, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 247, 254, 261, 268, 275, 282, 289, 296, 303, 310, 317, 300, 309, 318, 327, 336, 345, 354, 363, 372, 381, 390, 399, 360, 371, 382, 393, 404, 415, 426, 437, 448, 459, 470, 481, 420, 433, 446, 459, 472, 485, 498, 511, 524, 537, 550, 563};


    bulk_test( X,  mat, mode, unit_test);
}

// TEST(Test, ttm_test_3)
void test3()
{
    typedef Genten::DefaultHostExecutionSpace Space;
    typedef Genten::TensorT<Space> Tensor_type;
    typedef Genten::FacMatrixT<Space> FacMatrix_type;

    Genten::IndxArray tensor_dims;
    tensor_dims = Genten::IndxArray(4);
    tensor_dims[0] = 3;
    tensor_dims[1] = 4;
    tensor_dims[2] = 2;
    tensor_dims[3] = 2;

    Genten::IndxArray matrix_dims;
    matrix_dims = Genten::IndxArray(2);
    matrix_dims[0] = 5;
    matrix_dims[1] = 2;

    Tensor_type X(tensor_dims, 0.0);

    for (ttb_real i = 0; i < X.size().prod(); ++i)
    {
        X[i] = i;
    }

    Tensor_type mat(matrix_dims, 0.0);

    for (ttb_real i = 0; i < mat.size().prod(); ++i)
    {
        mat[i] = i;
    }
    ttb_indx mode = 3;

    Genten::IndxArrayT<Space> result_size(X.ndims());
    deep_copy(result_size, X.size());
    result_size[mode] = mat.size(0);

    Tensor_type Z(result_size, 0.0);
    std::cout << "preparing to calculate ttm:  5x2 * 3x4x2x2 along mode 3" << std::endl;

    int prod = Z.size().prod();

    ttb_real unit_test[120] = {120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225,
                               230, 235, 144, 151, 158, 165, 172, 179, 186, 193, 200, 207, 214, 221, 228, 235, 242, 249, 256, 263, 270, 277,
                               284, 291, 298, 305, 168, 177, 186, 195, 204, 213, 222, 231, 240, 249, 258, 267, 276, 285, 294, 303, 312, 321,
                               330, 339, 348, 357, 366, 375, 192, 203, 214, 225, 236, 247, 258, 269, 280, 291, 302, 313, 324, 335, 346, 357,
                               368, 379, 390, 401, 412, 423, 434, 445, 216, 229, 242, 255, 268, 281, 294, 307, 320, 333, 346, 359, 372, 385,
                               398, 411, 424, 437, 450, 463, 476, 489, 502, 515};

    bulk_test( X,  mat, mode, unit_test);

    // matrix_dims[1] = 3;
    // Tensor_type mat2(matrix_dims, 0.0);
    // Genten::Impl::genten_ttm_parfor_dgemm(4, X, mat, Z);
    // Genten::Impl::genten_ttm_serial_dgemm(1, X, mat, Z);
    // FacMatrix_type facmat(mat.size(0),mat.size(1),mat.getValues().values().data());
    // bulk_test( X, facmat, mode, unit_test);
}

void Genten_Test_TTM(int infolevel)
{
    initialize("Tests on Genten::TTM", infolevel);
    std::cout << "hello world" << std::endl;
    test0();
    test1();
    test2();
    test3();
    // testing::InitGoogleTest(&argc, argv);
    // Kokkos::initialize(argc, argv);
    // int ret = RUN_ALL_TESTS();
    // Kokkos::finalize();

    finalize();
}
