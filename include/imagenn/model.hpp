#ifndef IMAGENN_MODEL_HPP
#define IMAGENN_MODEL_HPP

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "imagenn/network.hpp"
#include "imagenn/spatial.hpp"
#include "imagenn/tensor.hpp"

/// @file model.hpp
/// @brief Модель: свёрточный экстрактор признаков перед полносвязной сетью.

namespace imagenn {

/// @brief Обучающий пример со спатиальным входом: изображение и эталонный выход.
using SpatialExample = std::pair<Tensor, std::vector<double>>;

/// @brief Модель из спатиальных слоёв (свёртка/подвыборка/выпрямление) и
/// полносвязной сети после них.
class Model {
  public:
    /// @brief Добавляет спатиальный слой в экстрактор признаков.
    /// @param layer слой свёртки, подвыборки или выпрямления
    void add_spatial(std::unique_ptr<SpatialLayer> layer);

    /// @brief Доступ к полносвязной части для построения и сохранения.
    /// @return ссылка на полносвязную сеть
    NeuralNetwork& dense();

    /// @brief Доступ к полносвязной части (только чтение).
    /// @return ссылка на полносвязную сеть
    const NeuralNetwork& dense() const;

    /// @brief Прогоняет изображение через всю модель.
    /// @param input входной тензор
    void run(const Tensor& input);

    /// @brief Обучает модель на наборе примеров за один проход.
    ///
    /// Порядок примеров перемешивается на каждом проходе: иначе данные,
    /// отсортированные по классам, приводят к катастрофическому забыванию и
    /// предсказанию одного класса.
    /// @param data обучающие примеры
    /// @param speed скорость обучения
    /// @param verbose выводить ли потерю по каждому примеру
    /// @param clip_value порог ограничения дельты веса
    /// @param use_clip применять ли ограничение
    /// @return суммарная потеря по всем примерам
    double train(const std::vector<SpatialExample>& data, double speed, bool verbose = false,
                 double clip_value = 5.0, bool use_clip = true);

    /// @brief Возвращает выходы модели.
    /// @return вектор выходов последнего слоя
    std::vector<double> get_output() const;

    /// @brief Возвращает индекс наибольшего выхода.
    /// @return индекс предсказанного класса
    int get_best_index() const;

    /// @brief Экспортирует параметры спатиальных слоёв.
    /// @return параметры по слоям (пустые для слоёв без весов)
    std::vector<std::vector<double>> export_spatial() const;

    /// @brief Загружает параметры спатиальных слоёв.
    /// @param data параметры по слоям в порядке слоёв
    /// @throws ValidationError если число слоёв не совпадает
    void import_spatial(const std::vector<std::vector<double>>& data);

    /// @brief Число спатиальных слоёв.
    /// @return количество слоёв экстрактора признаков
    std::size_t spatial_count() const { return feature_layers_.size(); }

  private:
    /// @brief Прогоняет вход через спатиальные слои.
    /// @param input входной тензор
    /// @return выход последнего спатиального слоя
    Tensor forward_features(const Tensor& input);

    std::vector<std::unique_ptr<SpatialLayer>> feature_layers_; ///< Свёрточная часть.
    NeuralNetwork dense_;                                       ///< Полносвязная часть.
};

} // namespace imagenn

#endif // IMAGENN_MODEL_HPP
