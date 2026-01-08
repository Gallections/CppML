cmake_minimum_required(VERSION 3.14)
project(CppML)

# 1. Setup - Use C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Enable Testing functionality in CMake
enable_testing()

# 3. Fetch GoogleTest (Downloads it automatically)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

# Force Google Test to use the same runtime library as your project
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Suppress annoying installation messages
set(INSTALL_GTEST OFF) 
FetchContent_MakeAvailable(googletest)

# 4. Handle Source Files
# Tell CMake where header files (.h/.hpp) live
include_directories(include)

# Find all .cpp files in src/
file(GLOB SOURCES "src/*.cpp")

# --- CORE CHANGE ---
# We create a "Static Library" named CppML_Lib.
# This compiles your math code once, so it can be reused.
add_library(CppML_Lib STATIC ${SOURCES})

# 5. Create the Main Executable (The App)
# We create the .exe, and link it to your library
add_executable(CppML src/CppML.cpp) # Assuming you have a main.cpp!
target_link_libraries(CppML PRIVATE CppML_Lib)

# 6. Setup Tests
if(EXISTS "${CMAKE_SOURCE_DIR}/tests")
    file(GLOB TEST_SOURCES "tests/*.cpp")
    if(TEST_SOURCES)
        add_executable(runTests ${TEST_SOURCES})
        
        # Link GTest (for the testing framework)
        # AND Link CppML_Lib (so the tests can see your Matrix code)
        target_link_libraries(runTests GTest::gtest_main CppML_Lib)
        
        target_include_directories(runTests PRIVATE include)

        # Register tests so they show up in Visual Studio Test Explorer
        include(GoogleTest)
        gtest_discover_tests(runTests)
    endif()
endif()