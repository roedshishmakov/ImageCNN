#ifndef IMAGENN_CONFIG_HPP
#define IMAGENN_CONFIG_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "imagenn/model.hpp"

/// \file config.hpp
/// \brief Конфигурация архитектуры сети и параметров обучения.

namespace imagenn {

/// \brief Описание одного слоя.
///
/// Тип слоя задаётся полем type ("dense", "conv", "maxpool", "flatten");
/// используются те поля, что относятся к данному типу.
struct LayerConfig {
    std::string type;           ///< Тип слоя.
    std::size_t size = 0;       ///< Число нейронов (dense).
    std::string activation;     ///< Имя функции активации (dense, conv).
    bool use_bias = false;      ///< Использовать ли смещение (dense).
    double random_radius = 0.1; ///< Радиус инициализации весов (dense, conv).
    int filters = 0;            ///< Число фильтров (conv).
    int kernel = 0;             ///< Сторона ядра (conv).
    int pool = 0;               ///< Сторона окна подвыборки (maxpool).
};

/// \brief Параметры обучения.
struct TrainingConfig {
    double learning_rate = 0.1;    ///< Скорость обучения.
    int epochs = 10;               ///< Число эпох.
    double clip_value = 5.0;       ///< Порог ограничения дельты веса.
    bool use_cross_entropy = true; ///< Использовать ли кросс-энтропию.
};

/// \brief Полная конфигурация сети.
struct NetworkConfig {
    std::vector<LayerConfig> layers; ///< Слои сети.
    TrainingConfig training;         ///< Параметры обучения.
};

/// \brief Загружает конфигурацию из файла.
///
/// Метка порядка байтов (BOM), которую добавляют некоторые редакторы, в начале
/// файла отбрасывается.
/// \param[in] path путь к файлу конфигурации
/// \return разобранная конфигурация
/// \throw PathError если файл не найден
/// \throw ValidationError при ошибке формата
NetworkConfig parse_config_file(const std::string& path);

/// \brief Сохраняет конфигурацию в файл.
/// \param[in] config конфигурация для сохранения
/// \param[in] path путь к файлу
/// \throw PathError если файл не удаётся открыть на запись
void save_config(const NetworkConfig& config, const std::string& path);

/// \brief Создаёт файл-шаблон конфигурации со стандартной архитектурой.
/// \param[in] path путь к создаваемому файлу
/// \throw PathError если файл не удаётся открыть на запись
void create_config_template(const std::string& path);

/// \brief Возвращает конфигурацию стандартной архитектуры.
/// \return конфигурация по умолчанию
NetworkConfig default_config();

/// \brief Строит модель по конфигурации.
///
/// Свёрточные слои, подвыборка и выпрямление образуют экстрактор признаков; за
/// ними следуют полносвязные слои. Если слой выпрямления не указан, он
/// добавляется автоматически перед первым полносвязным слоем.
/// \param[in] config конфигурация сети
/// \return построенная модель
/// \throw ValidationError если тип слоя неизвестен или порядок слоёв недопустим
Model build_model(const NetworkConfig& config);

} // namespace imagenn

#endif // IMAGENN_CONFIG_HPP
