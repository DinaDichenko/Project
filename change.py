import os

# Путь к папке с изображениями
path = 'C:\\Users\\super\\OneDrive\\Рабочий стол\\DI\\ВУЗЬ\\Воронкин х3\\ПРОЕКТ\\ПИКСЕЛЬ-АРТ'

# Получаем список файлов в папке
files = os.listdir(path)

# Сортируем файлы, чтобы обеспечить правильный порядок
files.sort()

# Счетчик для переименования файлов
counter = 1

# Перебираем файлы и переименовываем их
for file in files:
    # Получаем полный путь к файлу
    old_path = os.path.join(path, file)
    
    # Формируем новое имя файла (например, 2_1.jpg, 2_2.jpg и так далее)
    new_name = f'4_{counter}' + os.path.splitext(file)[1]
    
    # Формируем полный путь к новому файлу
    new_path = os.path.join(path, new_name)
    
    # Переименовываем файл
    os.rename(old_path, new_path)
    
    # Увеличиваем счетчик
    counter += 1