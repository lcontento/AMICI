project(unittests)

# cannot mix CppuTest new override togeter with Boost
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_OLD}")

add_executable(
    ${PROJECT_NAME}
    ../main.cpp
    tests1.cpp
    )

target_link_libraries(${PROJECT_NAME}
    amici-testing
    Upstream::amici
    )

find_package(Boost COMPONENTS serialization)
if(Boost_FOUND)
    target_sources(${PROJECT_NAME} PRIVATE testsSerialization.cpp)
    target_link_libraries(${PROJECT_NAME} Boost::serialization)
endif()

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME unittests COMMAND ./unittests -c)
