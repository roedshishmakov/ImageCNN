#ifndef IMAGENN_SPATIAL_HPP
#define IMAGENN_SPATIAL_HPP

#include <vector>

#include "imagenn/activations.hpp"
#include "imagenn/tensor.hpp"

/// \file spatial.hpp
/// \brief Спатиальные слои: свёртка, подвыборка и выпрямление.

namespace imagenn {

/// \brief Базовый класс спатиального слоя, работающего с тензорами.
///
/// Хранит форму входа и выхода. Слой кэширует данные прямого прохода, чтобы
/// использовать их при обратном.
class SpatialLayer {
  public:
    virtual ~SpatialLayer() = default;

    /// \brief Прямой проход.
    /// \param[in] input входной тензор
    /// \return выходной тензор
    /// \throw ValidationError если форма входа не совпадает с ожидаемой
    virtual Tensor forward(const Tensor& input) = 0;

    /// \brief Обратный проход.
    /// \param[in] grad_output градиент по выходу слоя
    /// \return градиент по входу слоя
    virtual Tensor backward(const Tensor& grad_output) = 0;

    /// \brief Применяет накопленные градиенты весов.
    /// \param[in] speed скорость обучения
    /// \param[in] clip_value порог ограничения обновления
    /// \param[in] use_clip применять ли ограничение
    virtual void apply(double speed, double clip_value, bool use_clip);

    /// \brief Экспортирует обучаемые параметры слоя.
    /// \return параметры слоя (пустой вектор для слоёв без весов)
    virtual std::vector<double> export_weights() const;

    /// \brief Загружает обучаемые параметры слоя.
    /// \param[in] weights параметры в порядке export_weights()
    virtual void import_weights(const std::vector<double>& weights);

    /// \brief Число каналов выхода.
    /// \return каналы выхода
    int out_channels() const { return out_c_; }

    /// \brief Высота выхода.
    /// \return высота выхода
    int out_height() const { return out_h_; }

    /// \brief Ширина выхода.
    /// \return ширина выхода
    int out_width() const { return out_w_; }

  protected:
    int in_c_ = 0;  ///< Каналы входа.
    int in_h_ = 0;  ///< Высота входа.
    int in_w_ = 0;  ///< Ширина входа.
    int out_c_ = 0; ///< Каналы выхода.
    int out_h_ = 0; ///< Высота выхода.
    int out_w_ = 0; ///< Ширина выхода.
};

/// \brief Свёрточный слой (аналог Conv2D), шаг 1, без дополнения краёв.
class ConvLayer : public SpatialLayer {
  public:
    /// \brief Создаёт свёрточный слой.
    /// \param[in] in_channels число входных каналов
    /// \param[in] in_height высота входа
    /// \param[in] in_width ширина входа
    /// \param[in] filters число фильтров (каналов выхода)
    /// \param[in] kernel сторона квадратного ядра
    /// \param[in] activation функция активации (применяется поэлементно)
    /// \param[in] random_radius радиус равномерной инициализации весов
    /// \throw ValidationError если параметры некорректны
    ConvLayer(int in_channels, int in_height, int in_width, int filters, int kernel,
              const ActivationBase& activation, double random_radius);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void apply(double speed, double clip_value, bool use_clip) override;
    std::vector<double> export_weights() const override;
    void import_weights(const std::vector<double>& weights) override;

    /// \brief Возвращает веса всех фильтров (для сохранения модели).
    /// \return веса в порядке (фильтр, канал, строка, столбец)
    const std::vector<double>& weights() const { return weights_; }

    /// \brief Возвращает смещения фильтров (для сохранения модели).
    /// \return смещения по одному на фильтр
    const std::vector<double>& biases() const { return biases_; }

    /// \brief Загружает веса и смещения.
    /// \param[in] weights веса в том же порядке, что и weights()
    /// \param[in] biases смещения по одному на фильтр
    /// \throw ValidationError если размеры не совпадают
    void load(const std::vector<double>& weights, const std::vector<double>& biases);

  private:
    /// \brief Индекс веса в плоском массиве.
    /// \param[in] f фильтр
    /// \param[in] c канал
    /// \param[in] ky строка ядра
    /// \param[in] kx столбец ядра
    /// \return индекс в weights_
    std::size_t weight_index(int f, int c, int ky, int kx) const;

    int kernel_ = 0;                   ///< Сторона ядра.
    const ActivationBase* act_;        ///< Функция активации.
    std::vector<double> weights_;      ///< Веса ядер.
    std::vector<double> biases_;       ///< Смещения фильтров.
    Tensor last_input_;                ///< Вход прямого прохода (для backward).
    std::vector<double> last_z_;       ///< Значения до активации (для backward).
    std::vector<double> weight_grads_; ///< Градиенты весов.
    std::vector<double> bias_grads_;   ///< Градиенты смещений.
};

/// \brief Слой подвыборки по максимуму (аналог MaxPooling2D).
class MaxPoolLayer : public SpatialLayer {
  public:
    /// \brief Создаёт слой подвыборки.
    /// \param[in] in_channels число каналов
    /// \param[in] in_height высота входа
    /// \param[in] in_width ширина входа
    /// \param[in] pool сторона окна (и шаг)
    /// \throw ValidationError если параметры некорректны
    MaxPoolLayer(int in_channels, int in_height, int in_width, int pool);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

  private:
    int pool_ = 0;                    ///< Сторона окна.
    std::vector<std::size_t> argmax_; ///< Индексы максимумов во входе (для backward).
};

/// \brief Слой выпрямления: переводит тензор в плоский вектор (1, 1, N).
class FlattenLayer : public SpatialLayer {
  public:
    /// \brief Создаёт слой выпрямления.
    /// \param[in] in_channels число каналов
    /// \param[in] in_height высота входа
    /// \param[in] in_width ширина входа
    FlattenLayer(int in_channels, int in_height, int in_width);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};

} // namespace imagenn

#endif // IMAGENN_SPATIAL_HPP
