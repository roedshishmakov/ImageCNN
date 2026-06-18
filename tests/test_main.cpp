#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "imagenn/activations.hpp"
#include "imagenn/exceptions.hpp"
#include "imagenn/network.hpp"
#include "imagenn/rng.hpp"

using namespace imagenn;

namespace {
NeuralNetwork make_small_network() {
    NeuralNetwork nn;
    nn.add_input_layer(2);
    nn.add_layer(3, sigmoid_activation(), 0.5, true);
    nn.add_layer(2, softmax_activation(), 0.5, false);
    return nn;
}
} // namespace

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

TEST_CASE("neuron loss is zero on a match and positive otherwise") {
    CHECK(Neuron::get_loss(1.0, 1.0) == doctest::Approx(0.0));
    CHECK(Neuron::get_loss(1.0, 0.0) == doctest::Approx(0.5));
    CHECK(Neuron::get_loss(0.0, 1.0) > 0.0);
}

TEST_CASE("forward pass through a softmax output produces a distribution") {
    set_random_seed(1);
    NeuralNetwork nn = make_small_network();
    nn.run({1.0, 0.0});
    const std::vector<double> out = nn.get_output();
    REQUIRE(out.size() == 2);
    CHECK(std::accumulate(out.begin(), out.end(), 0.0) == doctest::Approx(1.0));
}

TEST_CASE("run rejects an input of the wrong size") {
    NeuralNetwork nn = make_small_network();
    CHECK_THROWS_AS(nn.run({1.0}), ValidationError);
}

TEST_CASE("training reduces the loss on a separable problem") {
    set_random_seed(7);
    NeuralNetwork nn = make_small_network();
    const std::vector<TrainingExample> data = {{{1.0, 0.0}, {1.0, 0.0}}, {{0.0, 1.0}, {0.0, 1.0}}};
    const double first = nn.train(data, 0.5);
    for (int i = 0; i < 80; ++i) {
        nn.train(data, 0.5);
    }
    const double last = nn.train(data, 0.5);
    CHECK(last < first);
}

TEST_CASE("training rejects targets of the wrong size") {
    set_random_seed(9);
    NeuralNetwork nn = make_small_network();
    const std::vector<TrainingExample> bad = {{{1.0, 0.0}, {1.0, 0.0, 0.0}}};
    CHECK_THROWS_AS(nn.train(bad, 0.5), ValidationError);
}

TEST_CASE("exported weights restore an identical network") {
    set_random_seed(3);
    NeuralNetwork source = make_small_network();
    const std::vector<double> input = {0.7, 0.2};
    source.run(input);
    const std::vector<double> expected = source.get_output();

    set_random_seed(999);
    NeuralNetwork target = make_small_network();
    target.import_weights(source.export_weights());
    target.run(input);
    const std::vector<double> restored = target.get_output();

    REQUIRE(restored.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(restored[i] == doctest::Approx(expected[i]));
    }
}

TEST_CASE("import rejects a model with a different number of layers") {
    NeuralNetwork nn = make_small_network();
    std::vector<std::vector<std::vector<double>>> broken = {{}};
    CHECK_THROWS_AS(nn.import_weights(broken), ValidationError);
}

TEST_CASE("import rejects a layer of the wrong size") {
    set_random_seed(13);
    NeuralNetwork source = make_small_network();
    auto weights = source.export_weights();
    weights[1].push_back({});
    NeuralNetwork target = make_small_network();
    CHECK_THROWS_AS(target.import_weights(weights), ValidationError);
}

TEST_CASE("output queries reject an empty network") {
    NeuralNetwork empty;
    CHECK_THROWS_AS(empty.get_output(), ValidationError);
    CHECK_THROWS_AS(empty.get_best_index(), ValidationError);
}

TEST_CASE("best index points to the largest output") {
    set_random_seed(5);
    NeuralNetwork nn = make_small_network();
    nn.run({1.0, 0.0});
    const std::vector<double> out = nn.get_output();
    const int best = nn.get_best_index();
    REQUIRE(best >= 0);
    CHECK(out[static_cast<std::size_t>(best)] ==
          doctest::Approx(*std::max_element(out.begin(), out.end())));
}
