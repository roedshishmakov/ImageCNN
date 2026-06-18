# ImageCNN

Свёрточная нейросеть-классификатор изображений (рукописных цифр), реализованная
**с нуля на C++** без использования ML-фреймворков.

Сеть состоит из свёрточной части (Conv2D, MaxPooling, Flatten) и полносвязной
сети после неё. Модель обучается на пользовательском датасете, сохраняется в файл
и затем используется для классификации новых изображений.

## Возможности

- Свёрточные слои (Conv2D), подвыборка по максимуму (MaxPooling), выпрямление (Flatten).
- Полносвязные слои произвольной конфигурации.
- Функции активации: Transparent, ReLU, Sigmoid, Softmax.
- Выходной слой softmax с обучением по кросс-энтропии.
- Обучение с нуля, дообучение сохранённой модели.
- Сохранение и загрузка модели в собственном текстовом формате.
- График потерь в виде псевдографики в терминале.
- Классификация изображений из пользовательской папки.
- Настраиваемые пути ко всем рабочим папкам.

## Требования

- CMake 3.16 или новее.
- Компилятор с поддержкой C++17 (Apple Clang, GCC, MSVC).
- Доступ в интернет при первой конфигурации: CMake скачивает зависимости
  `doctest` и `stb` через FetchContent.

## Сборка

Linux / macOS:

```bash
cmake -S . -B build
cmake --build build
```

Windows (Visual Studio):

```bat
cmake -S . -B build
cmake --build build --config Release
```

Исполняемый файл: `build/imagecnn` (на Windows — `build/Release/imagecnn.exe`).

## Запуск тестов

```bash
ctest --test-dir build --output-on-failure
```

## Документация кода

Документация API собрана Doxygen в папке `docs/html` (открыть `docs/html/index.html`).
Пересобрать:

```bash
doxygen Doxyfile
# либо через CMake, если Doxygen найден при конфигурации:
cmake --build build --target docs
```

## Использование

Общий формат: `imagecnn <команда> [аргументы] [опции путей]`.

| Команда | Назначение |
|---|---|
| `--train, -t <model> <config> <dataset>` | обучить по файлу конфигурации |
| `--simple-train, -s <model> <epochs> <dataset>` | обучить стандартную архитектуру |
| `--fine, -f <model> <epochs> <dataset>` | дообучить сохранённую модель |
| `--load, -l <model> <images>` | классифицировать изображения |
| `--show-loss, -sl <model>` | вывести историю потерь |
| `--create-config, -c <name>` | создать шаблон конфигурации |
| `--graph, -g` | дополнительно вывести график потерь |
| `--help, -h` | справка |

Опции путей переопределяют рабочие папки (по умолчанию относительные):
`--configs-dir` (`configs`), `--weights-dir` (`weight_saves`), `--loss-dir` (`loss_saves`).

## Готовая модель

В папке `models/` лежит предобученная свёрточная модель `cnn` (обучена на MNIST,
точность около 97% на отложенной выборке):

```bash
imagecnn --load cnn <папка_с_изображениями> \
         --configs-dir models --weights-dir models --loss-dir models
```

Изображения должны быть рукописными цифрами в стиле MNIST; на картинках другого
стиля точность будет ниже.

## Сценарии использования

Быстрое обучение и проверка:

```bash
imagecnn --simple-train demo 10 examples
imagecnn --show-loss demo
imagecnn --load demo examples
```

Обучение по своей конфигурации:

```bash
imagecnn --create-config my_config
# отредактировать configs/my_config.config, затем:
imagecnn --train my_model my_config dataset/train
imagecnn --load my_model dataset/test
```

Дообучение:

```bash
imagecnn --fine my_model 15 dataset/extra
```

## Формат конфигурации

Текстовый файл `<name>.config` задаёт слои по порядку, затем параметры обучения.

Свёрточная сеть:

```ini
conv:8:3:relu:0.1
maxpool:2
flatten
dense:64:relu:true:0.1
dense:10:softmax:false:0.1

learning_rate=0.05
epochs=20
clip_value=5.0
use_cross_entropy=true
```

Форматы строк слоёв:

- `dense:[size]:[activation]:[use_bias]:[random_radius]` — полносвязный слой.
- `conv:[filters]:[kernel]:[activation]:[random_radius]` — свёрточный слой (шаг 1, без дополнения краёв).
- `maxpool:[size]` — подвыборка по максимуму с окном size×size.
- `flatten` — выпрямление карт признаков в вектор.

`activation` — `relu`, `sigmoid`, `softmax` или `transparent` (для `conv` — кроме `softmax`).
Вход — изображение 16×16 в оттенках серого, нормированное в [0, 1]. Свёрточные слои и
подвыборка идут до выпрямления, полносвязные — после. Если строка `flatten` не указана,
она добавляется автоматически перед первым `dense`.

## Формат датасета

- Формат файлов: PNG или JPG, размер любой (приводится к 16×16).
- Имя файла: `<метка>_<идентификатор>.<расширение>`, где `<метка>` — цифра 0–9.

```
dataset/
├── 0_image1.png
├── 1_image1.png
└── ...
```

## Структура проекта

```
ImageCNN/
├── CMakeLists.txt        сборка: библиотека + исполняемый файл + тесты
├── include/imagenn/      заголовки (публичный API, документация Doxygen)
├── src/                  реализации и точка входа (main.cpp)
├── tests/                тесты doctest, запускаются через ctest
├── examples/             примеры изображений
├── models/               предобученная модель
└── docs/                 документация Doxygen (HTML)
```

## Обработка ошибок

Все ошибки сообщаются через исключения и обрабатываются на верхнем уровне: при
ошибке программа печатает сообщение в stderr и возвращает код 1, при успехе — 0.
