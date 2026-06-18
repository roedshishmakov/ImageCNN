#include "imagenn/config.hpp"

#include <fstream>
#include <memory>
#include <sstream>

#include "imagenn/activations.hpp"
#include "imagenn/dataset.hpp"
#include "imagenn/exceptions.hpp"

namespace imagenn {

namespace {

std::string trim(const std::string& text) {
    const std::size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const std::size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        parts.push_back(item);
    }
    return parts;
}

bool to_bool(const std::string& value) {
    return trim(value) == "true";
}

const ActivationBase& activation_by_name(const std::string& name) {
    if (name == "relu") {
        return relu_activation();
    }
    if (name == "sigmoid") {
        return sigmoid_activation();
    }
    if (name == "softmax") {
        return softmax_activation();
    }
    if (name == "transparent") {
        return transparent_activation();
    }
    throw ValidationError("Unknown activation function: " + name);
}

LayerConfig parse_layer_line(const std::string& line, std::size_t line_number) {
    const std::vector<std::string> parts = split(line, ':');
    LayerConfig layer;
    layer.type = trim(parts[0]);

    try {
        if (layer.type == "dense") {
            if (parts.size() != 5) {
                throw ValidationError("Invalid dense layer at line " + std::to_string(line_number));
            }
            layer.size = static_cast<std::size_t>(std::stoul(trim(parts[1])));
            layer.activation = trim(parts[2]);
            layer.use_bias = to_bool(parts[3]);
            layer.random_radius = std::stod(trim(parts[4]));
        } else if (layer.type == "conv") {
            if (parts.size() != 5) {
                throw ValidationError("Invalid conv layer at line " + std::to_string(line_number));
            }
            layer.filters = std::stoi(trim(parts[1]));
            layer.kernel = std::stoi(trim(parts[2]));
            layer.activation = trim(parts[3]);
            layer.random_radius = std::stod(trim(parts[4]));
        } else if (layer.type == "maxpool") {
            if (parts.size() != 2) {
                throw ValidationError("Invalid maxpool layer at line " +
                                      std::to_string(line_number));
            }
            layer.pool = std::stoi(trim(parts[1]));
        } else if (layer.type == "flatten") {
            if (parts.size() != 1) {
                throw ValidationError("Invalid flatten layer at line " +
                                      std::to_string(line_number));
            }
        } else {
            throw ValidationError("Unknown layer type at line " + std::to_string(line_number));
        }
    } catch (const std::invalid_argument&) {
        throw ValidationError("Invalid layer values at line " + std::to_string(line_number));
    } catch (const std::out_of_range&) {
        throw ValidationError("Invalid layer values at line " + std::to_string(line_number));
    }
    return layer;
}

void parse_training_line(const std::string& line, TrainingConfig& training) {
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) {
        return;
    }
    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    try {
        if (key == "learning_rate") {
            training.learning_rate = std::stod(value);
        } else if (key == "epochs") {
            training.epochs = std::stoi(value);
        } else if (key == "clip_value") {
            training.clip_value = std::stod(value);
        } else if (key == "use_cross_entropy") {
            training.use_cross_entropy = to_bool(value);
        }
    } catch (const std::exception&) {
        throw ValidationError("Invalid training value for key: " + key);
    }
}

} // namespace

NetworkConfig parse_config_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw PathError("Config file not found: " + path);
    }

    NetworkConfig config;
    bool in_training_section = false;
    std::string raw;
    std::size_t line_number = 0;

    while (std::getline(file, raw)) {
        ++line_number;
        if (line_number == 1 && raw.size() >= 3 && static_cast<unsigned char>(raw[0]) == 0xEF &&
            static_cast<unsigned char>(raw[1]) == 0xBB &&
            static_cast<unsigned char>(raw[2]) == 0xBF) {
            raw.erase(0, 3);
        }
        const std::string line = trim(raw);

        if (line.empty() || line[0] == '#') {
            if (!config.layers.empty() && !in_training_section) {
                in_training_section = true;
            }
            continue;
        }

        if (!in_training_section) {
            config.layers.push_back(parse_layer_line(line, line_number));
        } else {
            parse_training_line(line, config.training);
        }
    }

    if (config.layers.empty()) {
        throw ValidationError("Configuration contains no layers");
    }
    return config;
}

void save_config(const NetworkConfig& config, const std::string& path) {
    std::ofstream file(path);
    if (!file) {
        throw PathError("Cannot open config file for writing: " + path);
    }

    file << "# Конфигурация нейронной сети\n\n";
    for (const LayerConfig& layer : config.layers) {
        if (layer.type == "conv") {
            file << "conv:" << layer.filters << ":" << layer.kernel << ":" << layer.activation
                 << ":" << layer.random_radius << "\n";
        } else if (layer.type == "maxpool") {
            file << "maxpool:" << layer.pool << "\n";
        } else if (layer.type == "flatten") {
            file << "flatten\n";
        } else {
            file << "dense:" << layer.size << ":" << layer.activation << ":"
                 << (layer.use_bias ? "true" : "false") << ":" << layer.random_radius << "\n";
        }
    }

    file << "\n# Параметры обучения\n";
    file << "learning_rate=" << config.training.learning_rate << "\n";
    file << "epochs=" << config.training.epochs << "\n";
    file << "clip_value=" << config.training.clip_value << "\n";
    file << "use_cross_entropy=" << (config.training.use_cross_entropy ? "true" : "false") << "\n";
}

void create_config_template(const std::string& path) {
    save_config(default_config(), path);
}

NetworkConfig default_config() {
    NetworkConfig config;
    config.layers.push_back({"dense", 32, "relu", true, 0.1});
    config.layers.push_back({"dense", 32, "relu", true, 0.1});
    config.layers.push_back({"dense", 10, "softmax", false, 0.1});
    return config;
}

Model build_model(const NetworkConfig& config) {
    Model model;
    int channels = 1;
    int height = kImageSize;
    int width = kImageSize;
    bool flattened = false;
    bool dense_started = false;
    int flat_size = 0;

    for (const LayerConfig& layer : config.layers) {
        if (layer.type == "conv") {
            if (flattened) {
                throw ValidationError("Convolution layer must precede flatten and dense layers");
            }
            if (layer.activation == "softmax") {
                throw ValidationError("Convolution layer cannot use softmax activation");
            }
            std::unique_ptr<ConvLayer> conv = std::make_unique<ConvLayer>(
                channels, height, width, layer.filters, layer.kernel,
                activation_by_name(layer.activation), layer.random_radius);
            channels = conv->out_channels();
            height = conv->out_height();
            width = conv->out_width();
            model.add_spatial(std::move(conv));
        } else if (layer.type == "maxpool") {
            if (flattened) {
                throw ValidationError("Pooling layer must precede flatten and dense layers");
            }
            std::unique_ptr<MaxPoolLayer> pool =
                std::make_unique<MaxPoolLayer>(channels, height, width, layer.pool);
            channels = pool->out_channels();
            height = pool->out_height();
            width = pool->out_width();
            model.add_spatial(std::move(pool));
        } else if (layer.type == "flatten") {
            if (flattened) {
                throw ValidationError("Duplicate flatten layer");
            }
            model.add_spatial(std::make_unique<FlattenLayer>(channels, height, width));
            flat_size = channels * height * width;
            flattened = true;
        } else if (layer.type == "dense") {
            if (!flattened) {
                model.add_spatial(std::make_unique<FlattenLayer>(channels, height, width));
                flat_size = channels * height * width;
                flattened = true;
            }
            if (!dense_started) {
                model.dense().add_input_layer(static_cast<std::size_t>(flat_size));
                dense_started = true;
            }
            model.dense().add_layer(layer.size, activation_by_name(layer.activation),
                                    layer.random_radius, layer.use_bias);
        } else {
            throw ValidationError("Unknown layer type: " + layer.type);
        }
    }

    if (!dense_started) {
        throw ValidationError("Configuration has no dense layers");
    }
    return model;
}

} // namespace imagenn
