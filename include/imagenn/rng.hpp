#ifndef IMAGENN_RNG_HPP
#define IMAGENN_RNG_HPP

/// \file rng.hpp
/// \brief Генератор псевдослучайных чисел для инициализации весов.

namespace imagenn {

/// \brief Задаёт начальное состояние генератора.
/// \param[in] seed зерно генератора
void set_random_seed(unsigned int seed);

/// \brief Возвращает случайное число из равномерного распределения на [lo, hi].
/// \param[in] lo нижняя граница
/// \param[in] hi верхняя граница
/// \return случайное значение в диапазоне [lo, hi]
double random_uniform(double lo, double hi);

} // namespace imagenn

#endif // IMAGENN_RNG_HPP
