add_executable(sandbox sandbox.cpp)
target_link_libraries(sandbox PUBLIC torchtb_static)
target_precompile_headers(sandbox PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/detail/pch.h>"
)
target_include_directories(sandbox PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:/include>
)

add_executable(floatcsv2parquet floatcsv2parquet.cpp)
target_link_libraries(floatcsv2parquet PUBLIC torchtb_static)
target_precompile_headers(floatcsv2parquet PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/detail/pch.h>"
)
target_include_directories(floatcsv2parquet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:/include>
)
