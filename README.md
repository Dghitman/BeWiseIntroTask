# BeWiseIntroTask

## Описание

На вход принимается файл 'test_data.csv'

На выход выдаются файлы: 
* 'processed_data.csv' - изначальная таблица с доп столбцами, которые помечают наличие приветствия, прощания, представления, а также имя в представлении и название компании.
* 'demand_check.csv' - маленькая таблица, в которой для каждого диалога помечено, отвечает ли он требованию о наличии приветствия и прщания.

## Алгоритм

Перед обработкой каждое слово приводится в начальную форму с помощью pymorphy

### 1. Приветствия
Для поиска приветствий использовано рег выражение: `'\b' + приветствие + '\b'`
Искомые приветствия:

`['здравствуйте', 'здравствовать', 'приветствовать', 'привет', 'добрый утро', "добрый день", "добрый вечер", "добрый ночь", "добрый здоровье", "добрый время сутки", "утро добрый", "день добрый", "вечер добрый", "ночь добрый"]`

### 2. Прощания
Для поиска прощаний использовано рег выражение: `'\b' + прощание + '\b'`
Искомые прощания:

`["до свидание", "до встреча", "все хороший", "все добрый", "все наилучший", "всего хороший", "всего добрый", "всего наилучший", "весь хороший", "весь добрый", "весь наилучший"]`

### 3. Представление
Для поиска представлений использовано рег выражение: `'\b' + представление + '\b'`, после которого должно идти слово (или последовательность имен), которое (которые) распознано(ы) pymorphy, как имееющее(ие) тэг 'Name'.
Искомые представления:

`["я", "я звать", "мой имя"]`

### 4. Компания

Для распознавания компаний не удалось найти достаточно хорошую готовую модель. Я пробовал NER из natasha, deeppavlov и spacy, но они все обучены, видимо, на данных, где текст представлен в литературном формате, т.е где имена и названия с большой буквы, присутствует пунктуация и т.п. Ни одна из моделей не находила названия компаний достаточно хорошо. (прим. "Яндекс" распознавался как компания, а "яндекс" - нет)

Syntax parser тоже не распознавал нормально, так что решил остановиться на
простом и не точном варианте.

Конечный алгоритм просто ищет слово "компания" в репликах, где менеджер представляется и берет следующее слово в качестве названия компании.

Такой алгоритм едва ли можно назвать хорошим, однако я не смог найти модели, позволяющие сделать хорошее распознавание, или придумать, как с найденными моделями иначе распознать названия компаний.
