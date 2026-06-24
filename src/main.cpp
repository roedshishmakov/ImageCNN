#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "imagenn/config.hpp"
#include "imagenn/dataset.hpp"
#include "imagenn/exceptions.hpp"
#include "imagenn/model.hpp"
#include "imagenn/model_io.hpp"
#include "imagenn/plot.hpp"
#include "imagenn/version.hpp"

/// \file main.cpp
/// \brief Интерфейс командной строки приложения.

namespace {

using namespace imagenn;

/// \brief Базовые папки для файлов модели, конфигурации и истории потерь.
struct CliOptions {
    std::string configs_dir = "configs"; ///< Папка с конфигурациями.
    std::string weights_dir = "models";  ///< Папка с сохранёнными моделями.
    std::string loss_dir = "loss_saves"; ///< Папка с историей потерь.
};

void print_logo() {
    std::cout << "+================================================+\n"
                 "|  ImageCNN  -  convolutional image classifier   |\n"
                 "+================================================+\n";
    std::cout << "version " << project_version() << "\n\n";
}

void print_help() {
    std::cout << "Usage: imagecnn <command> [args] [path options]\n\n"
                 "Commands:\n"
                 "  --train, -t <model> <config> <dataset>   train with a config file\n"
                 "  --simple-train, -s <model> <epochs> <dataset>  train default architecture\n"
                 "  --fine, -f <model> <epochs> <dataset>    continue training a saved model\n"
                 "  --load, -l <model> <images>              classify images with a saved model\n"
                 "  --show-loss, -sl <model>                 print the loss history\n"
                 "  --create-config, -c <name>               write a config template\n"
                 "  --graph, -g                              also print the loss graph\n"
                 "  --help, -h                               show this help\n\n"
                 "Path options (override default folders):\n"
                 "  --configs-dir <dir>   default: configs\n"
                 "  --weights-dir <dir>   default: models\n"
                 "  --loss-dir <dir>      default: loss_saves\n";
}

std::string join_path(const std::string& dir, const std::string& file) {
    return dir + "/" + file;
}

std::string config_path(const CliOptions& opt, const std::string& name) {
    return join_path(opt.configs_dir, name + ".config");
}

std::string model_path(const CliOptions& opt, const std::string& name) {
    return join_path(opt.weights_dir, name + ".nn");
}

std::string loss_path(const CliOptions& opt, const std::string& name) {
    return join_path(opt.loss_dir, name + ".txt");
}

void ensure_parent_dir(const std::string& path) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

bool contains(const std::vector<std::string>& flags, const std::string& flag) {
    return std::find(flags.begin(), flags.end(), flag) != flags.end();
}

bool has_command(const std::vector<std::string>& flags, const std::string& short_form,
                 const std::string& long_form) {
    return contains(flags, short_form) || contains(flags, long_form);
}

int parse_epochs(const std::string& text, int max_epochs) {
    int epochs = 0;
    try {
        epochs = std::stoi(text);
    } catch (const std::exception&) {
        throw EpochError("Epochs must be an integer, got '" + text + "'");
    }
    if (epochs <= 0) {
        throw EpochError("Number of epochs must be positive");
    }
    if (epochs > max_epochs) {
        throw EpochError("Number of epochs is too high");
    }
    return epochs;
}

void require_existing_path(const std::string& path, const std::string& description) {
    if (!std::filesystem::exists(path)) {
        throw PathError(description + " does not exist: " + path);
    }
}

/// \brief Обучает модель и сохраняет её, конфигурацию и историю потерь.
void train_and_save(Model& model, const NetworkConfig& config,
                    const std::vector<SpatialExample>& data, const CliOptions& opt,
                    const std::string& name) {
    std::vector<double> history;
    for (int epoch = 0; epoch < config.training.epochs; ++epoch) {
        const double loss = model.train(data, config.training.learning_rate, false,
                                        config.training.clip_value, true);
        history.push_back(loss);
        std::cout << "EPOCH #" << (epoch + 1) << "/" << config.training.epochs << " LOSS: " << loss
                  << "\n";
    }

    const std::string model_file = model_path(opt, name);
    const std::string loss_file = loss_path(opt, name);
    ensure_parent_dir(model_file);
    ensure_parent_dir(loss_file);
    save_model(model, model_file);
    save_losses(history, loss_file);
    std::cout << "Model saved: " << model_file << "\n";
}

void command_create_config(const std::vector<std::string>& args, const CliOptions& opt) {
    const std::string path = config_path(opt, args[0]);
    ensure_parent_dir(path);
    create_config_template(path);
    std::cout << "Config template created: " << path << "\n";
}

void command_show_loss(const std::vector<std::string>& args, const CliOptions& opt) {
    show_loss_ascii(load_losses(loss_path(opt, args[0])), std::cout);
}

void command_train(const std::vector<std::string>& args, const CliOptions& opt, bool graph) {
    const std::string& name = args[0];
    NetworkConfig config = parse_config_file(config_path(opt, args[1]));

    Model model = build_model(config);
    const std::string saved_config = config_path(opt, name);
    ensure_parent_dir(saved_config);
    save_config(config, saved_config);

    const std::vector<SpatialExample> data = load_training_examples(args[2]);
    std::cout << "Loaded " << data.size() << " training examples\n";

    train_and_save(model, config, data, opt, name);
    if (graph) {
        show_loss_ascii(load_losses(loss_path(opt, name)), std::cout);
    }
}

void command_simple_train(const std::vector<std::string>& args, const CliOptions& opt, bool graph) {
    const std::string& name = args[0];
    NetworkConfig config = default_config();
    config.training.epochs = parse_epochs(args[1], 10000);

    Model model = build_model(config);
    const std::string saved_config = config_path(opt, name);
    ensure_parent_dir(saved_config);
    save_config(config, saved_config);

    const std::vector<SpatialExample> data = load_training_examples(args[2]);
    std::cout << "Loaded " << data.size() << " training examples\n";

    train_and_save(model, config, data, opt, name);
    if (graph) {
        show_loss_ascii(load_losses(loss_path(opt, name)), std::cout);
    }
}

void command_fine(const std::vector<std::string>& args, const CliOptions& opt, bool graph) {
    const std::string& name = args[0];
    const int additional_epochs = parse_epochs(args[1], 1000);

    const std::string config_file = config_path(opt, name);
    NetworkConfig config =
        std::filesystem::exists(config_file) ? parse_config_file(config_file) : default_config();
    config.training.learning_rate = 0.01;
    config.training.epochs = additional_epochs;

    Model model = build_model(config);
    load_model(model, model_path(opt, name));

    const std::vector<SpatialExample> data = load_training_examples(args[2]);
    std::cout << "Loaded " << data.size() << " examples for fine-tuning\n";

    train_and_save(model, config, data, opt, name);
    if (graph) {
        show_loss_ascii(load_losses(loss_path(opt, name)), std::cout);
    }
}

void command_load(const std::vector<std::string>& args, const CliOptions& opt, bool graph) {
    const std::string& name = args[0];
    const std::string config_file = config_path(opt, name);

    Model model = build_model(std::filesystem::exists(config_file) ? parse_config_file(config_file)
                                                                   : default_config());
    load_model(model, model_path(opt, name));

    if (graph) {
        show_loss_ascii(load_losses(loss_path(opt, name)), std::cout);
    }

    const std::vector<NamedImage> images = load_images(args[1]);
    for (const NamedImage& sample : images) {
        model.run(sample.image);
        std::cout << "Test file: " << sample.name << "  Answer: " << model.get_best_index() << "\n";
    }
}

/// \brief Делит токены на опции путей, флаги и позиционные аргументы.
void parse_tokens(const std::vector<std::string>& tokens, CliOptions& opt,
                  std::vector<std::string>& flags, std::vector<std::string>& args) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];
        if (token == "--configs-dir" || token == "--weights-dir" || token == "--loss-dir") {
            if (i + 1 >= tokens.size()) {
                throw ArgumentError(token + " requires a path");
            }
            const std::string& value = tokens[++i];
            if (token == "--configs-dir") {
                opt.configs_dir = value;
            } else if (token == "--weights-dir") {
                opt.weights_dir = value;
            } else {
                opt.loss_dir = value;
            }
        } else if (!token.empty() && token[0] == '-') {
            flags.push_back(token);
        } else {
            args.push_back(token);
        }
    }
}

/// \brief Проверяет наличие команды, число аргументов и существование путей.
void validate_arguments(const std::vector<std::string>& flags, const std::vector<std::string>& args,
                        const CliOptions& opt) {
    const bool load = has_command(flags, "-l", "--load");
    const bool train = has_command(flags, "-t", "--train");
    const bool simple = has_command(flags, "-s", "--simple-train");
    const bool fine = has_command(flags, "-f", "--fine");
    const bool show_loss = has_command(flags, "-sl", "--show-loss");
    const bool create = has_command(flags, "-c", "--create-config");

    const int commands = load + train + simple + fine + show_loss + create;
    if (commands > 1) {
        throw IncorrectCommand("Choose only one main command");
    }

    if (train) {
        if (args.size() < 3) {
            throw ArgumentError("--train requires: model, config, dataset");
        }
        require_existing_path(config_path(opt, args[1]), "Config file");
        require_existing_path(args[2], "Training dataset");
    } else if (simple) {
        if (args.size() < 3) {
            throw ArgumentError("--simple-train requires: model, epochs, dataset");
        }
        parse_epochs(args[1], 10000);
        require_existing_path(args[2], "Training dataset");
    } else if (fine) {
        if (args.size() < 3) {
            throw ArgumentError("--fine requires: model, epochs, dataset");
        }
        parse_epochs(args[1], 1000);
        require_existing_path(args[2], "Dataset");
        require_existing_path(model_path(opt, args[0]), "Model");
    } else if (load) {
        if (args.size() < 2) {
            throw ArgumentError("--load requires: model, images");
        }
        require_existing_path(args[1], "Images path");
    } else if (show_loss) {
        if (args.empty()) {
            throw ArgumentError("--show-loss requires: model");
        }
        require_existing_path(loss_path(opt, args[0]), "Loss file");
    } else if (create) {
        if (args.empty()) {
            throw ArgumentError("--create-config requires: name");
        }
    }
}

int run(const std::vector<std::string>& tokens) {
    CliOptions opt;
    std::vector<std::string> flags;
    std::vector<std::string> args;
    parse_tokens(tokens, opt, flags, args);

    if (flags.empty() || has_command(flags, "-h", "--help")) {
        print_help();
        return 0;
    }

    validate_arguments(flags, args, opt);
    const bool graph = has_command(flags, "-g", "--graph");

    if (has_command(flags, "-c", "--create-config")) {
        command_create_config(args, opt);
    } else if (has_command(flags, "-sl", "--show-loss")) {
        command_show_loss(args, opt);
    } else if (has_command(flags, "-t", "--train")) {
        command_train(args, opt, graph);
    } else if (has_command(flags, "-s", "--simple-train")) {
        command_simple_train(args, opt, graph);
    } else if (has_command(flags, "-f", "--fine")) {
        command_fine(args, opt, graph);
    } else if (has_command(flags, "-l", "--load")) {
        command_load(args, opt, graph);
    } else {
        throw IncorrectCommand("Unknown command. Use --help for usage.");
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        print_logo();
        return run(std::vector<std::string>(argv + 1, argv + argc));
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
