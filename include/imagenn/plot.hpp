#ifndef IMAGENN_PLOT_HPP
#define IMAGENN_PLOT_HPP

#include <iosfwd>
#include <vector>

/// \file plot.hpp
/// \brief Вывод истории потерь в виде псевдографика.

namespace imagenn {

/// \brief Рисует историю потерь горизонтальными столбцами.
/// \param[in] losses значения потерь по эпохам
/// \param[out] out поток вывода
/// \throw ValidationError если список потерь пуст
void show_loss_ascii(const std::vector<double>& losses, std::ostream& out);

} // namespace imagenn

#endif // IMAGENN_PLOT_HPP
