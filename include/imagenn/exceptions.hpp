#ifndef IMAGENN_EXCEPTIONS_HPP
#define IMAGENN_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>

/// @file exceptions.hpp
/// @brief Иерархия пользовательских исключений приложения.

namespace imagenn {

/// @brief Базовое исключение для всех ошибок нейронной сети.
class NeuralNetworkError : public std::runtime_error {
  public:
    /// @brief Создаёт исключение с заданным сообщением.
    /// @param message текст ошибки
    explicit NeuralNetworkError(const std::string& message) : std::runtime_error(message) {}
};

/// @brief Ошибка в аргументах командной строки.
class ArgumentError : public NeuralNetworkError {
  public:
    /// @brief Создаёт исключение с заданным сообщением.
    /// @param message текст ошибки
    explicit ArgumentError(const std::string& message = "Invalid arguments provided")
        : NeuralNetworkError(message) {}
};

/// @brief Некорректная команда.
class IncorrectCommand : public NeuralNetworkError {
  public:
    /// @brief Создаёт исключение с заданным сообщением.
    /// @param message текст ошибки
    explicit IncorrectCommand(const std::string& message = "Command is incorrect")
        : NeuralNetworkError(message) {}
};

/// @brief Ошибка пути к файлу или папке.
class PathError : public NeuralNetworkError {
  public:
    /// @brief Создаёт исключение с заданным сообщением.
    /// @param message текст ошибки
    explicit PathError(const std::string& message = "Path error") : NeuralNetworkError(message) {}
};

/// @brief Ошибка в количестве эпох обучения.
class EpochError : public NeuralNetworkError {
  public:
    /// @brief Создаёт исключение с заданным сообщением.
    /// @param message текст ошибки
    explicit EpochError(const std::string& message = "Invalid number of epochs")
        : NeuralNetworkError(message) {}
};

/// @brief Ошибка валидации данных.
class ValidationError : public NeuralNetworkError {
  public:
    /// @brief Создаёт исключение с заданным сообщением.
    /// @param message текст ошибки
    explicit ValidationError(const std::string& message = "Validation error")
        : NeuralNetworkError(message) {}
};

} // namespace imagenn

#endif // IMAGENN_EXCEPTIONS_HPP
