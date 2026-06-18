#ifndef IMAGENN_VERSION_HPP
#define IMAGENN_VERSION_HPP

#include <string>

/// @file version.hpp
/// @brief Версия проекта.

/// @namespace imagenn
/// @brief Корневое пространство имён всех компонентов проекта.
namespace imagenn {

/// @brief Возвращает строку версии проекта.
/// @return семантическая версия, например "1.0.0"
std::string project_version();

} // namespace imagenn

#endif // IMAGENN_VERSION_HPP
