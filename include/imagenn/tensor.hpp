#ifndef IMAGENN_TENSOR_HPP
#define IMAGENN_TENSOR_HPP

#include <cstddef>
#include <vector>

/// \file tensor.hpp
/// \brief Трёхмерный тензор для спатиальных (свёрточных) слоёв.

namespace imagenn {

/// \brief Трёхмерный тензор (каналы, высота, ширина) с плоским хранением.
///
/// Данные лежат построчно: элемент (c, y, x) находится по индексу
/// ((c * height + y) * width + x).
struct Tensor {
    int channels = 0;         ///< Число каналов.
    int height = 0;           ///< Высота.
    int width = 0;            ///< Ширина.
    std::vector<double> data; ///< Значения, размер channels*height*width.

    /// \brief Создаёт пустой тензор.
    Tensor() = default;

    /// \brief Создаёт тензор заданной формы, заполненный нулями.
    /// \param[in] c число каналов
    /// \param[in] h высота
    /// \param[in] w ширина
    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w), data(static_cast<std::size_t>(c) * h * w, 0.0) {}

    /// \brief Возвращает плоский индекс элемента (c, y, x).
    /// \param[in] c канал
    /// \param[in] y строка
    /// \param[in] x столбец
    /// \return индекс в массиве data
    std::size_t index(int c, int y, int x) const {
        return (static_cast<std::size_t>(c) * height + y) * width + x;
    }

    /// \brief Читает элемент (c, y, x).
    /// \param[in] c канал
    /// \param[in] y строка
    /// \param[in] x столбец
    /// \return значение элемента
    double at(int c, int y, int x) const { return data[index(c, y, x)]; }

    /// \brief Доступ к элементу (c, y, x) для записи.
    /// \param[in] c канал
    /// \param[in] y строка
    /// \param[in] x столбец
    /// \return ссылка на элемент
    double& at(int c, int y, int x) { return data[index(c, y, x)]; }

    /// \brief Возвращает общее число элементов.
    /// \return channels*height*width
    int size() const { return channels * height * width; }
};

} // namespace imagenn

#endif // IMAGENN_TENSOR_HPP
