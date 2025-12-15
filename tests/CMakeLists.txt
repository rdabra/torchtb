
#################################################### Test Executable ####################################################

find_package(GTest REQUIRED)

add_executable(
  torchtb_tests
  torchtb_tests.cpp
  tCSV_IO.cpp
  tParquet_IO.cpp
  tAnalyticTable.cpp
  tConverter.cpp
  tXYMatrix.cpp
  tAnalyticTableNumeric.cpp
  tTrainingBundle.cpp
)  

target_precompile_headers(torchtb_tests PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/detail/pch.h>"
)

target_link_libraries(
  torchtb_tests  
  GTest::gtest GTest::gtest_main torchtb_static 
)


include(GoogleTest)
gtest_discover_tests(torchtb_tests)

