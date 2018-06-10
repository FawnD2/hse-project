# Учебный проект по индексации лиц на видео 

### Требования

  * Python 3.3+ или Python 2.7
  * macOS или Linux (Windows официально не поддерживается используемыми библиотеками, но может работать) 
  * dlib с модулем для Python ([Как установить dlib на MacOS или Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf))
  * opencv
  ```bash
pip3 install opencv-python
```
  * модуль face_recognition через

```bash
pip3 install face_recognition
```
 
### Использование

Данный проект решает задачу индексации лиц на видео из файла и с вебкамеры. По умолчанию база для текущего видео пуста, но это можно изменить при запуске. Для использования нужно запустить файл face_detection.py. 
При запуске требуется указать некоторые параметры [параметры по умолчанию]:

#### Enter video file [Webcam]
Полный путь к видео для обработки. Чтобы обработать файл из текущей директории достаточно просто указать его название.

#### Enter sensitivity [0.65]
Порог распознавания (чувствительность детектора). Варьируется от 0 до 1. Чем ниже параметр, тем строже детектор реагирует на изменения.

#### Enter scale [2]
Масштабируемость видео. Детектор сжимает обрабатываемые кадры в scale раз. Увеличение параметра снизит качество распознавания, но увеличит скорость обработки.

#### Enter framerate [2]
Частота обработки кадров. Рассмотривается каждый framerate кадр. Увеличение параметра приведет к снижению частоты обработки кадров (прорисовываемые рамки будут отставать от видео), но увеличит скорость.

#### Preload images from directory [Don't]
Путь к директории из которой будут загружены фотографии для изначальной базы лиц. По умолчанию база пуста. Поддерживаются только изображения с разрешениями .jpg и .png. Имена распознаваемым людям из базы будут совпадать с названиями файлов.

#### Record video [n]
Выберите опцию "y" чтобы обработанное видео с прорисованными рамками было записано. Оно сохранится в текущую директорию как "output.avi"  

