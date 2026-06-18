#ifndef IMAGENN_MODEL_IO_HPP
#define IMAGENN_MODEL_IO_HPP

#include <string>
#include <vector>

#include "imagenn/model.hpp"
#include "imagenn/network.hpp"

/// @file model_io.hpp
/// @brief Сохранение и загрузка моделей и истории потерь.

namespace imagenn {

/// @brief Сохраняет веса сети в текстовый файл.
/// @param network сеть для сохранения
/// @param path путь к файлу модели
/// @throws PathError если файл не удаётся открыть на запись
void save_model(const NeuralNetwork& network, const std::string& path);

/// @brief Загружает веса в сеть из текстового файла.
/// @param network сеть, в которую загружаются веса
/// @param path путь к файлу модели
/// @throws PathError если файл не найден
/// @throws ValidationError при несоответствии формата или структуры сети
void load_model(NeuralNetwork& network, const std::string& path);

/// @brief Сохраняет модель (свёрточную и полносвязную части) в текстовый файл.
/// @param model модель для сохранения
/// @param path путь к файлу модели
/// @throws PathError если файл не удаётся открыть на запись
void save_model(const Model& model, const std::string& path);

/// @brief Загружает веса в модель из текстового файла.
/// @param model модель, в которую загружаются веса
/// @param path путь к файлу модели
/// @throws PathError если файл не найден
/// @throws ValidationError при несоответствии формата или структуры модели
void load_model(Model& model, const std::string& path);

/// @brief Сохраняет историю потерь в файл, по одному значению на строку.
/// @param losses значения потерь
/// @param path путь к файлу
/// @param append дописывать ли в конец файла вместо перезаписи
/// @throws PathError если файл не удаётся открыть на запись
void save_losses(const std::vector<double>& losses, const std::string& path, bool append = false);

/// @brief Загружает историю потерь из файла.
/// @param path путь к файлу
/// @return значения потерь
/// @throws PathError если файл не найден
/// @throws ValidationError если файл содержит некорректные данные
std::vector<double> load_losses(const std::string& path);

} // namespace imagenn

#endif // IMAGENN_MODEL_IO_HPP
