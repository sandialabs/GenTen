include(GoogleTest)

function(add_genten_test test_name)
    add_executable(${test_name} ${test_name}.cpp Unit_Tests_Main.cpp)
    target_link_libraries(${test_name} gentenlib)
    target_link_libraries(${test_name} gtest)
    # NOTE (ETP, 9/26/22):  gtest_discover_tests requires running the tests at
    # build time to add them to CTest, which may not work in all environments
    # (e.g., a Cuda build on a compile node that doesn't have any GPUs).  There
    # is a new DISCOVERY_MODE that can defer this to test launch, but it is
    # not clear how to make that work with MPI.
    #
    # Instead just add the tests like normal, which actually makes things run
    # a lot faster anyway.
    #gtest_discover_tests(${test_name} WORKING_DIRECTORY ${Genten_BINARY_DIR} DISCOVERY_MODE PRE_TEST)
    genten_add_test(NAME ${test_name} COMMAND "${Genten_BINARY_DIR}/bin/${test_name}" WORKING_DIRECTORY ${Genten_BINARY_DIR})
endfunction()

add_genten_test(Genten_Test_Array)
add_genten_test(Genten_Test_CpAls)
add_genten_test(Genten_Test_FacMatrix)
add_genten_test(Genten_Test_HessVec)
add_genten_test(Genten_Test_IndxArray)
add_genten_test(Genten_Test_IOtext)
add_genten_test(Genten_Test_Ktensor)
add_genten_test(Genten_Test_MixedFormats)
add_genten_test(Genten_Test_Sptensor)
add_genten_test(Genten_Test_Tensor)
add_genten_test(Genten_Test_TTM)

IF(ENABLE_LBFGSB)
add_genten_test(Genten_Test_CpOptLbfgsb)
ENDIF()

IF(ENABLE_ROL)
add_genten_test(Genten_Test_CpOptRol)
ENDIF()


IF(ENABLE_GCP)
add_genten_test(Genten_Test_GCP_SGD)
    # IF(ENABLE_ROL)
    #     add_genten_test(Genten_Test_GCP_Opt)
    # ENDIF()
ENDIF()
