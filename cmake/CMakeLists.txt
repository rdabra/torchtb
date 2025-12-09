include(GNUInstallDirs)


# Using alias
add_library(torchtb::torchtb_static ALIAS torchtb_static)
add_library(torchtb::torchtb_shared ALIAS torchtb_shared)
add_executable(torchtb::floatcsv2parquet ALIAS floatcsv2parquet)

# SOVERSION e VERSION para a .so
set_target_properties(torchtb_shared PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
  VERSION   ${PROJECT_VERSION})

# Avoiding name conflicts in windows
if (WIN32)
  set_target_properties(torchtb_static PROPERTIES OUTPUT_NAME "torchtb_static")
  set_target_properties(torchtb_shared PROPERTIES OUTPUT_NAME "torchtb")
endif()


# Headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Bins e libs
install(TARGETS
    thirdparty_libs
    torchtb_static
    torchtb_shared
    floatcsv2parquet
  EXPORT torchtbTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}          # exe + dll (Windows)
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}          # .so/.dylib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}          # .a e import libs
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  CONFIGURATIONS Release # Installation only in release
)

# Export dos targets
install(EXPORT torchtbTargets
  NAMESPACE torchtb::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchtb)

# Arquivos de pacote (Config/Version)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/torchtbConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/torchtbConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/torchtbConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchtb)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/torchtbConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/torchtbConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchtb)

# Opcional: RPATH para achar libs lado-a-lado após instalar
if(UNIX AND NOT APPLE)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
endif()
