#ifndef IMAGENN_NEURON_HPP
#define IMAGENN_NEURON_HPP

#include <vector>

#include "imagenn/activations.hpp"

/// @file neuron.hpp
/// @brief Нейрон и его специализированные варианты.

namespace imagenn {

class Link;

/// @brief Нейрон сети.
///
/// Накапливает входной сигнал, применяет функцию активации и хранит
/// не-владеющие ссылки на входящие и исходящие связи.
class Neuron {
  public:
    /// @brief Создаёт нейрон с заданной функцией активации.
    /// @param activation функция активации (хранится по ссылке, не копируется)
    explicit Neuron(const ActivationBase& activation) : activation_(&activation) {}

    virtual ~Neuron() = default;

    /// @brief Добавляет входящую связь.
    /// @param link указатель на связь
    void add_link_input(Link* link) { link_input_.push_back(link); }

    /// @brief Добавляет исходящую связь.
    /// @param link указатель на связь
    void add_link_output(Link* link) { link_output_.push_back(link); }

    /// @brief Подаёт сигнал на вход нейрона (накапливается).
    /// @param signal входной сигнал
    void add(double signal) { input_ += signal; }

    /// @brief Сбрасывает вход нейрона.
    /// @param input новое значение входа
    virtual void reset(double input = 0.0) { input_ = input; }

    /// @brief Вычисляет выход нейрона по накопленному входу.
    virtual void activation() { output_ = get_activation(input_); }

    /// @brief Применяет функцию активации к значению.
    /// @param x входное значение
    /// @return значение функции активации
    virtual double get_activation(double x) const { return activation_->calc(x); }

    /// @brief Применяет производную функции активации к значению.
    /// @param x входное значение
    /// @return значение производной
    virtual double get_activation_derivative(double x) const { return activation_->derivative(x); }

    /// @brief Передаёт выход нейрона по всем исходящим связям.
    void send() const;

    /// @brief Считает дельты весов входящих связей для выходного слоя.
    /// @param ref эталонное значение
    /// @param speed скорость обучения
    void back_propagation_output_l(double ref, double speed);

    /// @brief Считает дельты весов входящих связей для скрытого слоя.
    /// @param speed скорость обучения
    void back_propagation_hidden_l(double speed);

    /// @brief Применяет накопленные дельты к весам исходящих связей.
    void back_propagation_apply();

    /// @brief Экспортирует веса входящих связей.
    /// @return веса в порядке входящих связей
    std::vector<double> export_weights() const;

    /// @brief Импортирует веса входящих связей.
    /// @param data веса в порядке входящих связей
    void import_weights(const std::vector<double>& data);

    /// @brief Вычисляет квадратичную потерю.
    /// @param output выход нейрона
    /// @param ref эталонное значение
    /// @return значение потери 0.5 * (output - ref)^2
    static double get_loss(double output, double ref) {
        return 0.5 * (output - ref) * (output - ref);
    }

    /// @brief Возвращает накопленный вход нейрона.
    /// @return значение входа
    double input() const { return input_; }

    /// @brief Возвращает выход нейрона.
    /// @return значение выхода
    double output() const { return output_; }

    /// @brief Устанавливает выход нейрона напрямую.
    /// @param value новое значение выхода
    void set_output(double value) { output_ = value; }

    /// @brief Возвращает список входящих связей.
    /// @return ссылки на входящие связи
    const std::vector<Link*>& links_input() const { return link_input_; }

    /// @brief Возвращает список исходящих связей.
    /// @return ссылки на исходящие связи
    const std::vector<Link*>& links_output() const { return link_output_; }

  protected:
    const ActivationBase* activation_; ///< Функция активации нейрона.
    double input_ = 0.0;               ///< Накопленный вход.
    double output_ = 0.0;              ///< Текущий выход.
    std::vector<Link*> link_input_;    ///< Входящие связи (не-владеющие).
    std::vector<Link*> link_output_;   ///< Исходящие связи (не-владеющие).
};

/// @brief Прозрачный нейрон: выход равен входу. Используется во входном слое.
class TransparentNeuron : public Neuron {
  public:
    /// @brief Создаёт прозрачный нейрон.
    /// @param activation функция активации
    explicit TransparentNeuron(const ActivationBase& activation) : Neuron(activation) {}

    /// @brief Копирует вход в выход без изменений.
    void activation() override { output_ = input_; }
};

/// @brief Нейрон смещения: всегда выдаёт постоянный выход, равный 1.
class BiasNeuron : public Neuron {
  public:
    /// @brief Создаёт нейрон смещения.
    /// @param activation функция активации
    explicit BiasNeuron(const ActivationBase& activation) : Neuron(activation) {
        input_ = 1.0;
        output_ = 1.0;
    }

    /// @brief Игнорирует сброс: выход смещения остаётся постоянным.
    /// @param input не используется
    void reset(double input = 0.0) override { (void) input; }
};

/// @brief Нейрон выходного слоя softmax.
///
/// На своём уровне ведёт себя как тождество (выход равен входу); нормализация
/// softmax выполняется на уровне слоя.
class NeuronSoftmax : public Neuron {
  public:
    /// @brief Создаёт softmax-нейрон.
    /// @param activation функция активации
    explicit NeuronSoftmax(const ActivationBase& activation) : Neuron(activation) {}

    /// @brief Копирует вход (логит) в выход.
    void activation() override { output_ = input_; }

    /// @brief Возвращает значение без изменений.
    /// @param x входное значение
    /// @return то же значение x
    double get_activation(double x) const override { return x; }

    /// @brief Возвращает производную, равную 1.
    /// @param x входное значение
    /// @return всегда 1
    double get_activation_derivative(double x) const override {
        (void) x;
        return 1.0;
    }
};

} // namespace imagenn

#endif // IMAGENN_NEURON_HPP
