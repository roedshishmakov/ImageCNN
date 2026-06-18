#ifndef IMAGENN_LINK_HPP
#define IMAGENN_LINK_HPP

/// @file link.hpp
/// @brief Связь между двумя нейронами.

namespace imagenn {

class Neuron;

/// @brief Взвешенная связь, передающая сигнал от одного нейрона к другому.
class Link {
  public:
    /// @brief Создаёт связь между нейронами.
    /// @param from нейрон-источник
    /// @param to нейрон-приёмник
    /// @param weight начальный вес связи
    Link(Neuron& from, Neuron& to, double weight) : from_(&from), to_(&to), weight_(weight) {}

    /// @brief Передаёт сигнал приёмнику, умножая его на вес.
    /// @param signal передаваемый сигнал
    void send(double signal) const;

    /// @brief Считает дельту веса для связи, входящей в выходной слой.
    /// @param ref эталонное значение
    /// @param speed скорость обучения
    void back_propagation_output_l(double ref, double speed);

    /// @brief Считает дельту веса для связи, входящей в скрытый слой.
    /// @param speed скорость обучения
    void back_propagation_hidden_l(double speed);

    /// @brief Применяет накопленную дельту к весу.
    void apply_weight_delta();

    /// @brief Возвращает текущий вес связи.
    /// @return вес
    double weight() const { return weight_; }

    /// @brief Возвращает накопленную дельту веса.
    /// @return дельта веса
    double weight_delta() const { return weight_delta_; }

    /// @brief Возвращает параметр дельты, используемый при обратном проходе.
    /// @return параметр дельты
    double weight_delta_param() const { return weight_delta_param_; }

    /// @brief Устанавливает дельту веса.
    /// @param value новая дельта
    void set_weight_delta(double value) { weight_delta_ = value; }

    /// @brief Устанавливает параметр дельты.
    /// @param value новый параметр дельты
    void set_weight_delta_param(double value) { weight_delta_param_ = value; }

    /// @brief Возвращает выход нейрона-источника.
    /// @return выход источника
    double from_output() const;

    /// @brief Экспортирует вес связи.
    /// @return значение веса
    double export_weight() const { return weight_; }

    /// @brief Импортирует вес связи.
    /// @param value сохранённый вес
    void import_weight(double value) { weight_ = value; }

  private:
    Neuron* from_;                    ///< Нейрон-источник (не-владеющий).
    Neuron* to_;                      ///< Нейрон-приёмник (не-владеющий).
    double weight_;                   ///< Вес связи.
    double weight_delta_ = 0.0;       ///< Накопленная дельта веса.
    double weight_delta_param_ = 0.0; ///< Промежуточный параметр обратного прохода.
};

} // namespace imagenn

#endif // IMAGENN_LINK_HPP
