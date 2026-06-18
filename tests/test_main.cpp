#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "imagenn/version.hpp"

using namespace imagenn;

TEST_CASE("project version is set") {
    CHECK(project_version() == "1.0.0");
    CHECK_FALSE(project_version().empty());
}
