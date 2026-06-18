#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "imagenn/activations.hpp"
#include "imagenn/config.hpp"
#include "imagenn/dataset.hpp"
#include "imagenn/exceptions.hpp"
#include "imagenn/model.hpp"
#include "imagenn/model_io.hpp"
#include "imagenn/network.hpp"
#include "imagenn/plot.hpp"
#include "imagenn/rng.hpp"
#include "imagenn/spatial.hpp"
#include "imagenn/tensor.hpp"

using namespace imagenn;

namespace {
NeuralNetwork make_small_network() {
    NeuralNetwork nn;
    nn.add_input_layer(2);
    nn.add_layer(3, sigmoid_activation(), 0.5, true);
    nn.add_layer(2, softmax_activation(), 0.5, false);
    return nn;
}

std::string temp_path(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
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

TEST_CASE("tensor stores and reads elements by coordinates") {
    Tensor t(2, 3, 4);
    CHECK(t.size() == 24);
    t.at(1, 2, 3) = 7.0;
    CHECK(t.at(1, 2, 3) == doctest::Approx(7.0));
    CHECK(t.index(1, 2, 3) == 23);
}

TEST_CASE("convolution computes a known result and output shape") {
    ConvLayer conv(1, 3, 3, 1, 2, transparent_activation(), 0.0);
    conv.load({1.0, 1.0, 1.0, 1.0}, {0.0});
    Tensor input(1, 3, 3);
    for (double& v : input.data) {
        v = 1.0;
    }
    const Tensor out = conv.forward(input);
    CHECK(out.channels == 1);
    CHECK(out.height == 2);
    CHECK(out.width == 2);
    for (double v : out.data) {
        CHECK(v == doctest::Approx(4.0));
    }
}

TEST_CASE("convolution output dimensions follow filters and kernel") {
    ConvLayer conv(2, 5, 5, 3, 3, relu_activation(), 0.1);
    CHECK(conv.out_channels() == 3);
    CHECK(conv.out_height() == 3);
    CHECK(conv.out_width() == 3);
}

TEST_CASE("convolution rejects a wrong input shape and a too large kernel") {
    ConvLayer conv(1, 4, 4, 1, 2, relu_activation(), 0.1);
    Tensor wrong(1, 3, 3);
    CHECK_THROWS_AS(conv.forward(wrong), ValidationError);
    CHECK_THROWS_AS(ConvLayer(1, 2, 2, 1, 3, relu_activation(), 0.1), ValidationError);
}

TEST_CASE("convolution gradient matches a numeric estimate") {
    ConvLayer conv(1, 3, 3, 1, 2, transparent_activation(), 0.0);
    const std::vector<double> w0 = {0.1, -0.2, 0.3, 0.5};
    conv.load(w0, {0.0});
    Tensor input(1, 3, 3);
    for (int i = 0; i < input.size(); ++i) {
        input.data[static_cast<std::size_t>(i)] = 0.1 * (i + 1);
    }
    conv.forward(input);
    Tensor ones(conv.out_channels(), conv.out_height(), conv.out_width());
    for (double& v : ones.data) {
        v = 1.0;
    }
    conv.backward(ones);
    const std::vector<double> before = conv.weights();
    conv.apply(1.0, 0.0, false);
    const std::vector<double> after = conv.weights();

    const double eps = 1e-6;
    for (std::size_t i = 0; i < w0.size(); ++i) {
        std::vector<double> wp = w0;
        std::vector<double> wm = w0;
        wp[i] += eps;
        wm[i] -= eps;
        conv.load(wp, {0.0});
        const Tensor out_p = conv.forward(input);
        conv.load(wm, {0.0});
        const Tensor out_m = conv.forward(input);
        double sum_p = 0.0;
        double sum_m = 0.0;
        for (double v : out_p.data) {
            sum_p += v;
        }
        for (double v : out_m.data) {
            sum_m += v;
        }
        const double analytic = before[i] - after[i];
        const double numeric = (sum_p - sum_m) / (2.0 * eps);
        CHECK(analytic == doctest::Approx(numeric).epsilon(1e-4));
    }
}

TEST_CASE("max pooling takes the window maximum and routes the gradient back") {
    MaxPoolLayer pool(1, 2, 2, 2);
    Tensor input(1, 2, 2);
    input.data = {1.0, 2.0, 3.0, 4.0};
    const Tensor out = pool.forward(input);
    CHECK(out.at(0, 0, 0) == doctest::Approx(4.0));
    Tensor grad(1, 1, 1);
    grad.data = {5.0};
    const Tensor back = pool.backward(grad);
    CHECK(back.data[3] == doctest::Approx(5.0));
    CHECK(back.data[0] == doctest::Approx(0.0));
}

TEST_CASE("max pooling downsamples the spatial size") {
    MaxPoolLayer pool(1, 4, 4, 2);
    CHECK(pool.out_height() == 2);
    CHECK(pool.out_width() == 2);
}

TEST_CASE("flatten reshapes to a vector and back") {
    FlattenLayer flatten(2, 2, 2);
    CHECK(flatten.out_channels() == 1);
    CHECK(flatten.out_width() == 8);
    Tensor input(2, 2, 2);
    for (int i = 0; i < input.size(); ++i) {
        input.data[static_cast<std::size_t>(i)] = static_cast<double>(i);
    }
    const Tensor flat = flatten.forward(input);
    REQUIRE(flat.data.size() == 8);
    CHECK(flat.data[5] == doctest::Approx(5.0));
    const Tensor back = flatten.backward(flat);
    CHECK(back.channels == 2);
    CHECK(back.data[5] == doctest::Approx(5.0));
}

namespace {
Model make_conv_model() {
    Model model;
    model.add_spatial(std::make_unique<ConvLayer>(1, 6, 6, 2, 3, sigmoid_activation(), 0.2));
    model.add_spatial(std::make_unique<MaxPoolLayer>(2, 4, 4, 2));
    model.add_spatial(std::make_unique<FlattenLayer>(2, 2, 2));
    model.dense().add_input_layer(8);
    model.dense().add_layer(2, softmax_activation(), 0.5, false);
    return model;
}
} // namespace

TEST_CASE("model forward through convolution layers produces a distribution") {
    set_random_seed(1);
    Model model = make_conv_model();
    Tensor image(1, 6, 6);
    for (double& v : image.data) {
        v = 0.5;
    }
    model.run(image);
    const std::vector<double> out = model.get_output();
    REQUIRE(out.size() == 2);
    CHECK(std::accumulate(out.begin(), out.end(), 0.0) == doctest::Approx(1.0));
}

TEST_CASE("model training reduces the loss with convolution layers") {
    set_random_seed(7);
    Model model = make_conv_model();
    Tensor a(1, 6, 6);
    Tensor b(1, 6, 6);
    for (double& v : a.data) {
        v = 0.1;
    }
    for (double& v : b.data) {
        v = 0.9;
    }
    const std::vector<SpatialExample> data = {{a, {1.0, 0.0}}, {b, {0.0, 1.0}}};
    const double first = model.train(data, 0.1);
    for (int i = 0; i < 150; ++i) {
        model.train(data, 0.1);
    }
    const double last = model.train(data, 0.1);
    CHECK(last < first);
}

TEST_CASE("model weights restore an identical convolution network") {
    set_random_seed(3);
    Model source = make_conv_model();
    Tensor image(1, 6, 6);
    for (int i = 0; i < image.size(); ++i) {
        image.data[static_cast<std::size_t>(i)] = 0.01 * i;
    }
    source.run(image);
    const std::vector<double> expected = source.get_output();

    set_random_seed(999);
    Model target = make_conv_model();
    target.import_spatial(source.export_spatial());
    target.dense().import_weights(source.dense().export_weights());
    target.run(image);
    const std::vector<double> restored = target.get_output();

    REQUIRE(restored.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(restored[i] == doctest::Approx(expected[i]));
    }
}

TEST_CASE("config parsing reads layers and training parameters") {
    const std::string path = temp_path("imagecnn_cfg_ok.config");
    {
        std::ofstream file(path);
        file << "# network\ndense:32:relu:true:0.1\ndense:10:softmax:false:0.2\n";
        file << "\nlearning_rate=0.05\nepochs=7\nclip_value=3.0\nuse_cross_entropy=false\n";
    }
    const NetworkConfig config = parse_config_file(path);
    REQUIRE(config.layers.size() == 2);
    CHECK(config.layers[0].size == 32);
    CHECK(config.layers[0].activation == "relu");
    CHECK(config.layers[1].activation == "softmax");
    CHECK(config.training.epochs == 7);
    CHECK(config.training.learning_rate == doctest::Approx(0.05));
    CHECK(config.training.use_cross_entropy == false);
    std::filesystem::remove(path);
}

TEST_CASE("config parsing rejects a malformed layer line") {
    const std::string path = temp_path("imagecnn_cfg_bad.config");
    {
        std::ofstream file(path);
        file << "dense:32:relu:true\n";
    }
    CHECK_THROWS_AS(parse_config_file(path), ValidationError);
    std::filesystem::remove(path);
}

TEST_CASE("a convolution config survives a save/load round trip") {
    NetworkConfig config;
    config.layers.push_back({"conv", 0, "relu", false, 0.1, 8, 3, 0});
    config.layers.push_back({"maxpool", 0, "", false, 0.1, 0, 0, 2});
    config.layers.push_back({"flatten", 0, "", false, 0.1, 0, 0, 0});
    config.layers.push_back({"dense", 10, "softmax", false, 0.1, 0, 0, 0});
    const std::string path = temp_path("imagecnn_conv.config");
    save_config(config, path);
    const NetworkConfig restored = parse_config_file(path);
    REQUIRE(restored.layers.size() == 4);
    CHECK(restored.layers[0].type == "conv");
    CHECK(restored.layers[0].filters == 8);
    CHECK(restored.layers[1].pool == 2);
    CHECK(restored.layers[2].type == "flatten");
    std::filesystem::remove(path);
}

TEST_CASE("build_model builds the configured layers and rejects unknown activation") {
    const Model ok = build_model(default_config());
    CHECK(ok.dense().layer_count() == 4);
    CHECK(ok.spatial_count() == 1);
    NetworkConfig bad;
    bad.layers.push_back({"dense", 4, "bogus", false, 0.1, 0, 0, 0});
    CHECK_THROWS_AS(build_model(bad), ValidationError);
}

TEST_CASE("build_model builds a convolution model from config") {
    NetworkConfig config;
    config.layers.push_back({"conv", 0, "relu", false, 0.1, 4, 3, 0});
    config.layers.push_back({"maxpool", 0, "", false, 0.1, 0, 0, 2});
    config.layers.push_back({"flatten", 0, "", false, 0.1, 0, 0, 0});
    config.layers.push_back({"dense", 10, "softmax", false, 0.1, 0, 0, 0});
    const Model model = build_model(config);
    CHECK(model.spatial_count() == 3);
}

TEST_CASE("loss plot renders a header and rejects empty input") {
    std::ostringstream out;
    show_loss_ascii({1.0, 0.5, 0.25}, out);
    CHECK(out.str().find("Total loss") != std::string::npos);
    CHECK_THROWS_AS(show_loss_ascii({}, out), ValidationError);
}

TEST_CASE("training examples are loaded with one-hot targets") {
    const auto data = load_training_examples(IMAGENN_TEST_DATA_DIR);
    REQUIRE_FALSE(data.empty());
    for (const auto& example : data) {
        CHECK(example.first.size() == kInputSize);
        CHECK(example.second.size() == static_cast<std::size_t>(kNumClasses));
        CHECK(std::accumulate(example.second.begin(), example.second.end(), 0.0) ==
              doctest::Approx(1.0));
    }
}

TEST_CASE("image is converted to a normalized tensor") {
    const std::string image = std::string(IMAGENN_TEST_DATA_DIR) + "/0_1.png";
    const Tensor input = image_to_tensor(image);
    CHECK(input.channels == 1);
    CHECK(input.size() == kInputSize);
    double min_v = input.data[0];
    double max_v = input.data[0];
    for (double value : input.data) {
        CHECK(value >= 0.0);
        CHECK(value <= 1.0);
        if (value < min_v) {
            min_v = value;
        }
        if (value > max_v) {
            max_v = value;
        }
    }
    CHECK(max_v > min_v);
}

TEST_CASE("reading a missing image reports a path error") {
    CHECK_THROWS_AS(image_to_tensor(temp_path("imagecnn_no_image.png")), PathError);
}

TEST_CASE("images are loaded with file names") {
    const std::vector<NamedImage> images = load_images(IMAGENN_TEST_DATA_DIR);
    REQUIRE_FALSE(images.empty());
    for (const NamedImage& sample : images) {
        CHECK_FALSE(sample.name.empty());
        CHECK(sample.image.size() == kInputSize);
    }
}

TEST_CASE("loading from a missing directory reports a path error") {
    CHECK_THROWS_AS(load_images(temp_path("imagecnn_no_such_dir")), PathError);
}

TEST_CASE("a saved network reloads into an identical network") {
    set_random_seed(11);
    NeuralNetwork source = make_small_network();
    const std::vector<double> input = {0.3, 0.9};
    source.run(input);
    const std::vector<double> expected = source.get_output();
    const std::string path = temp_path("imagecnn_model.nn");
    save_model(source, path);

    set_random_seed(222);
    NeuralNetwork target = make_small_network();
    load_model(target, path);
    target.run(input);
    const std::vector<double> restored = target.get_output();
    REQUIRE(restored.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(restored[i] == doctest::Approx(expected[i]));
    }
    std::filesystem::remove(path);
}

TEST_CASE("loading a missing model reports a path error") {
    NeuralNetwork nn = make_small_network();
    CHECK_THROWS_AS(load_model(nn, temp_path("imagecnn_missing.nn")), PathError);
}

TEST_CASE("loading a corrupted model file reports a validation error") {
    const std::string path = temp_path("imagecnn_corrupt.nn");
    {
        std::ofstream file(path);
        file << "this is not a model\n";
    }
    NeuralNetwork nn = make_small_network();
    CHECK_THROWS_AS(load_model(nn, path), ValidationError);
    std::filesystem::remove(path);
}

TEST_CASE("loss history survives a save/load round trip and append") {
    const std::string path = temp_path("imagecnn_loss.txt");
    save_losses({0.1}, path, false);
    save_losses({0.2}, path, true);
    const std::vector<double> restored = load_losses(path);
    REQUIRE(restored.size() == 2);
    CHECK(restored[0] == doctest::Approx(0.1));
    CHECK(restored[1] == doctest::Approx(0.2));
    std::filesystem::remove(path);
}

TEST_CASE("loading missing loss history reports a path error") {
    CHECK_THROWS_AS(load_losses(temp_path("imagecnn_no_loss.txt")), PathError);
}

TEST_CASE("a saved convolution model reloads identically") {
    set_random_seed(11);
    Model source = make_conv_model();
    Tensor image(1, 6, 6);
    for (int i = 0; i < image.size(); ++i) {
        image.data[static_cast<std::size_t>(i)] = 0.02 * i;
    }
    source.run(image);
    const std::vector<double> expected = source.get_output();
    const std::string path = temp_path("imagecnn_conv_model.nn");
    save_model(source, path);

    set_random_seed(222);
    Model target = make_conv_model();
    load_model(target, path);
    target.run(image);
    const std::vector<double> restored = target.get_output();
    REQUIRE(restored.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(restored[i] == doctest::Approx(expected[i]));
    }
    std::filesystem::remove(path);
}

TEST_CASE("convolution constructor rejects non-positive parameters") {
    CHECK_THROWS_AS(ConvLayer(1, 4, 4, 0, 2, relu_activation(), 0.1), ValidationError);
    CHECK_THROWS_AS(ConvLayer(0, 4, 4, 2, 2, relu_activation(), 0.1), ValidationError);
}

TEST_CASE("convolution load and import reject mismatched sizes") {
    ConvLayer conv(1, 4, 4, 2, 3, relu_activation(), 0.1);
    CHECK_THROWS_AS(conv.load({1.0, 2.0}, {0.0, 0.0}), ValidationError);
    CHECK_THROWS_AS(conv.import_weights({1.0, 2.0}), ValidationError);
}

TEST_CASE("max pooling constructor rejects invalid windows") {
    CHECK_THROWS_AS(MaxPoolLayer(1, 4, 4, 0), ValidationError);
    CHECK_THROWS_AS(MaxPoolLayer(1, 2, 2, 3), ValidationError);
}

TEST_CASE("max pooling forward rejects a wrong input shape") {
    MaxPoolLayer pool(1, 4, 4, 2);
    Tensor wrong(1, 3, 3);
    CHECK_THROWS_AS(pool.forward(wrong), ValidationError);
}

TEST_CASE("flatten forward rejects a wrong input shape") {
    FlattenLayer flatten(2, 2, 2);
    Tensor wrong(1, 2, 2);
    CHECK_THROWS_AS(flatten.forward(wrong), ValidationError);
}

TEST_CASE("model rejects a wrong spatial import and a wrong image shape") {
    set_random_seed(2);
    Model model = make_conv_model();
    CHECK_THROWS_AS(model.import_spatial({}), ValidationError);
    Tensor wrong(1, 4, 4);
    CHECK_THROWS_AS(model.run(wrong), ValidationError);
}
