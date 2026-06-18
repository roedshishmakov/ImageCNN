#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "imagenn/config.hpp"
#include "imagenn/exceptions.hpp"
#include "imagenn/model_io.hpp"
#include "imagenn/plot.hpp"
#include "imagenn/version.hpp"

/// @file main.cpp
/// @brief Интерфейс командной строки приложения.

namespace {

using namespace imagenn;

/// @brief Базовые папки для файлов модели, конфигурации и истории потерь.
struct CliOptions {
    std::string configs_dir = "configs";      ///< Папка с конфигурациями.
    std::string weights_dir = "weight_saves"; ///< Папка с сохранёнными моделями.
    std::string loss_dir = "loss_saves";      ///< Папка с историей потерь.
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
                 "  --weights-dir <dir>   default: weight_saves\n"
                 "  --loss-dir <dir>      default: loss_saves\n";
}

std::string join_path(const std::string& dir, const std::string& file) {
    return dir + "/" + file;
}

std::string config_path(const CliOptions& opt, const std::string& name) {
    return join_path(opt.configs_dir, name + ".config");
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

void require_existing_path(const std::string& path, const std::string& description) {
    if (!std::filesystem::exists(path)) {
        throw PathError(description + " does not exist: " + path);
    }
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

/// @brief Делит токены на опции путей, флаги и позиционные аргументы.
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

int run(const std::vector<std::string>& tokens) {
    CliOptions opt;
    std::vector<std::string> flags;
    std::vector<std::string> args;
    parse_tokens(tokens, opt, flags, args);

    if (flags.empty() || has_command(flags, "-h", "--help")) {
        print_help();
        return 0;
    }

    if (has_command(flags, "-c", "--create-config")) {
        if (args.empty()) {
            throw ArgumentError("--create-config requires: name");
        }
        command_create_config(args, opt);
    } else if (has_command(flags, "-sl", "--show-loss")) {
        if (args.empty()) {
            throw ArgumentError("--show-loss requires: model");
        }
        require_existing_path(loss_path(opt, args[0]), "Loss file");
        command_show_loss(args, opt);
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
