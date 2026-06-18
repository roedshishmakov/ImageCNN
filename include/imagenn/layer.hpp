#ifndef IMAGENN_LAYER_HPP
#define IMAGENN_LAYER_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "imagenn/activations.hpp"
#include "imagenn/link.hpp"
#include "imagenn/neuron.hpp"

/// @file layer.hpp
/// @brief Слой нейронов и его softmax-вариант.

namespace imagenn {

/// @brief Тип нейронов в слое.
enum class NeuronType {
    Regular,     ///< Обычный нейрон.
    Transparent, ///< Прозрачный нейрон (входной слой).
    Softmax      ///< Нейрон выходного слоя softmax.
};

/// @brief Слой нейронов.
///
/// Владеет своими нейронами, нейроном смещения и связями смещения.
class Layer {
  public:
    /// @brief Создаёт слой заданного размера.
    ///
    /// Веса смещения инициализируются нулём: случайные смещения при активном bias
    /// заталкивают часть ReLU-нейронов в отрицательную зону и выключают их.
    /// @param type тип нейронов слоя
    /// @param size количество нейронов
    /// @param activation функция активации нейронов
    /// @param use_bias добавлять ли нейрон смещения
    Layer(NeuronType type, std::size_t size, const ActivationBase& activation, bool use_bias);

    virtual ~Layer() = default;

    /// @brief Сбрасывает входы всех нейронов в ноль.
    void reset();

    /// @brief Устанавливает входы нейронов заданными значениями.
    /// @param input значения входов; длина должна совпадать с размером слоя
    /// @throws ValidationError если длина input не равна размеру слоя
    void reset(const std::vector<double>& input);

    /// @brief Вычисляет выходы всех нейронов слоя.
    virtual void calc();

    /// @brief Считает суммарную потерю слоя.
    /// @param refs эталонные значения
    /// @return значение потери (квадратичная ошибка)
    /// @throws ValidationError если длина refs не равна размеру слоя
    virtual double get_loss(const std::vector<double>& refs) const;

    /// @brief Передаёт выходы всех нейронов по их связям.
    void send() const;

    /// @brief Считает дельты весов для выходного слоя.
    /// @param refs эталонные значения
    /// @param speed скорость обучения
    virtual void back_propagation_output_l(const std::vector<double>& refs, double speed);

    /// @brief Считает дельты весов для скрытого слоя.
    /// @param speed скорость обучения
    void back_propagation_hidden_l(double speed);

    /// @brief Применяет накопленные дельты весов слоя.
    void back_propagation_apply();

    /// @brief Экспортирует веса слоя.
    /// @return веса каждого нейрона
    std::vector<std::vector<double>> export_weights() const;

    /// @brief Импортирует веса слоя.
    /// @param data веса каждого нейрона
    /// @throws ValidationError если число нейронов не совпадает
    void import_weights(const std::vector<std::vector<double>>& data);

    /// @brief Возвращает число нейронов в слое.
    /// @return размер слоя
    std::size_t size() const { return neurons_.size(); }

    /// @brief Возвращает нейроны слоя.
    /// @return вектор нейронов
    std::vector<std::unique_ptr<Neuron>>& neurons() { return neurons_; }

    /// @brief Возвращает нейроны слоя (только чтение).
    /// @return вектор нейронов
    const std::vector<std::unique_ptr<Neuron>>& neurons() const { return neurons_; }

  protected:
    std::vector<std::unique_ptr<Neuron>> neurons_; ///< Нейроны слоя.
    std::unique_ptr<Neuron> bias_;                 ///< Нейрон смещения (может отсутствовать).
    std::vector<std::unique_ptr<Link>> links_;     ///< Связи смещения, принадлежащие слою.
};

/// @brief Выходной слой с функцией активации softmax и кросс-энтропийной потерей.
class LayerSoftmax : public Layer {
  public:
    /// @brief Создаёт softmax-слой заданного размера.
    /// @param size количество нейронов
    explicit LayerSoftmax(std::size_t size);

    /// @brief Вычисляет softmax по всем выходам слоя.
    void calc() override;

    /// @brief Считает кросс-энтропийную потерю.
    /// @param refs эталонное распределение
    /// @return значение потери
    /// @throws ValidationError если длина refs не равна размеру слоя
    double get_loss(const std::vector<double>& refs) const override;

    /// @brief Считает дельты весов выходного softmax-слоя.
    /// @param refs эталонное распределение
    /// @param speed скорость обучения
    void back_propagation_output_l(const std::vector<double>& refs, double speed) override;
};

} // namespace imagenn

#endif // IMAGENN_LAYER_HPP
