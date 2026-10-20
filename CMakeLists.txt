cmake_minimum_required(VERSION 3.15...3.30)
project(qrq LANGUAGES CXX)

# Mirror the reference build: g++ -O2 -std=c++17 -fPIC -fvisibility=hidden,
# without -DNDEBUG and without link-time optimisation.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# The translation unit includes cpp/qrq_matched_audit.inc from its own directory.
pybind11_add_module(_qrq_matched MODULE NO_EXTRAS cpp/trajectory_trees_3d.cpp)

install(TARGETS _qrq_matched LIBRARY DESTINATION qrq)
