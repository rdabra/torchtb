#################################################### Library Sources ####################################################

set(TORCHTB_SOURCES 
  CSV_IO.cpp
  Parquet_IO.cpp
  AnalyticTable.cpp
  AnalyticTableNumeric.cpp
  Converter.cpp
  XYMatrix.cpp
  detail/utils.cpp
)


#################################################### Dependecies ####################################################

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

# Aggregating dependencies in a single target 
add_library(thirdparty_libs INTERFACE)
target_link_libraries(thirdparty_libs INTERFACE ${TORCH_LIBRARIES} Parquet::parquet_shared Arrow::arrow_shared)


#################################################### Static Library ####################################################

add_library(torchtb_static STATIC ${TORCHTB_SOURCES})
target_precompile_headers(torchtb_static PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/detail/pch.h>"
)
target_include_directories(torchtb_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(torchtb_static PUBLIC thirdparty_libs)
set_target_properties(torchtb_static PROPERTIES OUTPUT_NAME "torchtb") 


#################################################### Dynamic Library ####################################################

set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)

add_library(torchtb_shared SHARED ${TORCHTB_SOURCES})
target_precompile_headers(torchtb_shared PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/detail/pch.h>"
)
target_compile_definitions(torchtb_shared PUBLIC torchtb_shared)
target_include_directories(torchtb_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(torchtb_shared PUBLIC thirdparty_libs)
set_target_properties(torchtb_shared PROPERTIES OUTPUT_NAME "torchtb")


