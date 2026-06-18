#ifndef IMAGENN_DATASET_HPP
#define IMAGENN_DATASET_HPP

#include <string>
#include <utility>
#include <vector>

#include "imagenn/tensor.hpp"

/// \file dataset.hpp
/// \brief Загрузка изображений в тензоры и подготовка обучающих данных.

namespace imagenn {

/// Сторона, к которой приводится изображение перед подачей в сеть.
constexpr int kImageSize = 16;
/// Число элементов изображения (для одноканального входа).
constexpr int kInputSize = kImageSize * kImageSize;
/// Число распознаваемых классов (цифры 0–9).
constexpr int kNumClasses = 10;

/// \brief Изображение вместе с именем файла.
struct NamedImage {
    std::string name; ///< Имя файла.
    Tensor image;     ///< Нормированный тензор изображения.
};

/// \brief Преобразует файл изображения в нормированный тензор 1×kImageSize×kImageSize.
///
/// Изображение приводится к оттенкам серого, масштабируется и нормируется в [0, 1].
/// \param[in] path путь к файлу изображения
/// \return тензор со значениями в диапазоне [0, 1]
/// \throw PathError если изображение не удаётся прочитать
Tensor image_to_tensor(const std::string& path);

/// \brief Загружает изображения из папки для распознавания.
/// \param[in] directory путь к папке с изображениями
/// \return изображения с именами файлов, отсортированные по имени
/// \throw PathError если папка не существует
std::vector<NamedImage> load_images(const std::string& directory);

/// \brief Загружает обучающие примеры из папки.
///
/// Метка класса берётся из имени файла до первого символа '_'.
/// \param[in] directory путь к папке с изображениями
/// \return пары (тензор изображения, one-hot эталон)
/// \throw PathError если папка не существует
/// \throw ValidationError если метку в имени файла не удаётся разобрать
std::vector<std::pair<Tensor, std::vector<double>>>
load_training_examples(const std::string& directory);

} // namespace imagenn

#endif // IMAGENN_DATASET_HPP
