#ifndef IMAGENN_NETWORK_HPP
#define IMAGENN_NETWORK_HPP

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "imagenn/activations.hpp"
#include "imagenn/layer.hpp"
#include "imagenn/link.hpp"

/// \file network.hpp
/// \brief Полносвязная нейронная сеть.

namespace imagenn {

/// \brief Обучающий пример: вектор входа и эталонный вектор выхода.
using TrainingExample = std::pair<std::vector<double>, std::vector<double>>;

/// \brief Полносвязная нейронная сеть из последовательных слоёв.
class NeuralNetwork {
  public:
    /// \brief Добавляет входной слой из прозрачных нейронов.
    /// \param[in] size количество входов
    void add_input_layer(std::size_t size);

    /// \brief Добавляет слой и связывает его с предыдущим.
    /// \param[in] size количество нейронов
    /// \param[in] activation функция активации (softmax создаёт выходной softmax-слой)
    /// \param[in] random_radius радиус равномерной инициализации весов
    /// \param[in] use_bias добавлять ли нейрон смещения
    void add_layer(std::size_t size, const ActivationBase& activation, double random_radius,
                   bool use_bias);

    /// \brief Прогоняет вход через сеть.
    /// \param[in] input входной вектор
    /// \throw ValidationError если сеть пуста или размер входа неверен
    void run(const std::vector<double>& input);

    /// \brief Обучает сеть на наборе примеров за один проход.
    /// \param[in] data обучающие примеры
    /// \param[in] speed скорость обучения
    /// \param[in] verbose выводить ли потерю по каждому примеру
    /// \param[in] clip_value порог ограничения дельты веса
    /// \param[in] use_clip применять ли ограничение дельты
    /// \return суммарная потеря по всем примерам
    double train(const std::vector<TrainingExample>& data, double speed, bool verbose = false,
                 double clip_value = 5.0, bool use_clip = true);

    /// \brief Возвращает градиент по значениям входного слоя.
    ///
    /// Действителен после обратного прохода и до применения дельт.
    /// \return градиент по каждому входу сети
    std::vector<double> input_gradient() const;

    /// \brief Выполняет обратный проход для одного примера; run() должен быть уже вызван.
    /// \param[in] target эталонный вектор
    /// \param[in] speed скорость обучения
    /// \param[in] clip_value порог ограничения дельты веса
    /// \param[in] use_clip применять ли ограничение
    /// \param[out] input_grad_out если не nullptr, сюда записывается градиент по входу
    /// \return потеря на этом примере
    double backward_step(const std::vector<double>& target, double speed, double clip_value,
                         bool use_clip, std::vector<double>* input_grad_out);

    /// \brief Экспортирует веса всех слоёв.
    /// \return веса по слоям и нейронам
    std::vector<std::vector<std::vector<double>>> export_weights() const;

    /// \brief Импортирует веса всех слоёв.
    /// \param[in] data веса по слоям и нейронам
    /// \throw ValidationError если структура данных не совпадает со структурой сети
    void import_weights(const std::vector<std::vector<std::vector<double>>>& data);

    /// \brief Возвращает выходы последнего слоя.
    /// \return вектор выходов
    /// \throw ValidationError если сеть пуста
    std::vector<double> get_output() const;

    /// \brief Возвращает индекс наибольшего выхода.
    /// \return индекс максимального выхода
    /// \throw ValidationError если сеть пуста
    int get_best_index() const;

    /// \brief Возвращает число слоёв сети.
    /// \return количество слоёв
    std::size_t layer_count() const { return layers_.size(); }

  private:
    /// \brief Полностью связывает два последних добавленных слоя.
    /// \param[in] random_radius радиус равномерной инициализации весов
    void connect_last_two(double random_radius);

    std::vector<std::unique_ptr<Layer>> layers_; ///< Слои сети.
    std::vector<std::unique_ptr<Link>> links_;   ///< Межслойные связи, принадлежащие сети.
};

} // namespace imagenn

#endif // IMAGENN_NETWORK_HPP
