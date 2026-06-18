#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <numeric>
#include <stdexcept>
#include <vector>

#include "imagenn/activations.hpp"

using namespace imagenn;

TEST_CASE("transparent activation returns the input") {
    ActivationTransparent a;
    CHECK(a.calc(2.5) == doctest::Approx(2.5));
    CHECK(a.calc(-2.5) == doctest::Approx(-2.5));
    CHECK(a.derivative(2.5) == doctest::Approx(1.0));
}

TEST_CASE("relu activation") {
    ActivationRelu a;
    CHECK(a.calc(3.0) == doctest::Approx(3.0));
    CHECK(a.calc(-3.0) == doctest::Approx(0.0));
    CHECK(a.derivative(3.0) == doctest::Approx(1.0));
    CHECK(a.derivative(-3.0) == doctest::Approx(0.0));
}

TEST_CASE("sigmoid activation") {
    ActivationSigmoid a;
    CHECK(a.calc(0.0) == doctest::Approx(0.5));
    CHECK(a.calc(20.0) == doctest::Approx(1.0).epsilon(0.001));
    CHECK(a.calc(-20.0) == doctest::Approx(0.0).epsilon(0.001));
    CHECK(a.derivative(0.0) == doctest::Approx(0.25));
    const double s = a.calc(2.0);
    CHECK(a.derivative(2.0) == doctest::Approx(s * (1.0 - s)));
}

TEST_CASE("softmax over a layer produces a probability distribution") {
    const std::vector<double> out = ActivationSoftmax::calc_layer({1.0, 2.0, 3.0});
    REQUIRE(out.size() == 3);
    CHECK(std::accumulate(out.begin(), out.end(), 0.0) == doctest::Approx(1.0));
    CHECK(out[2] > out[1]);
    CHECK(out[1] > out[0]);
}

TEST_CASE("softmax handles large logits without overflow") {
    const std::vector<double> out = ActivationSoftmax::calc_layer({200.0, 100.0});
    REQUIRE(out.size() == 2);
    CHECK(std::accumulate(out.begin(), out.end(), 0.0) == doctest::Approx(1.0));
    CHECK(out[0] > out[1]);
}

TEST_CASE("softmax rejects an empty layer") {
    CHECK_THROWS_AS(ActivationSoftmax::calc_layer({}), std::invalid_argument);
}

TEST_CASE("softmax scalar calls are not supported") {
    ActivationSoftmax a;
    CHECK_THROWS_AS(a.calc(1.0), std::logic_error);
    CHECK_THROWS_AS(a.derivative(1.0), std::logic_error);
}
