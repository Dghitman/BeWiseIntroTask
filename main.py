import pandas as pd
import re
import numpy as np

import pymorphy2

from nltk import word_tokenize

df_data = pd.read_csv('test_data.csv')

greetings = ['здравствуйте', 'здравствовать', 'приветствовать', 'привет', 'добрый утро', "добрый день", "добрый вечер", "добрый ночь",
             "добрый здоровье", "добрый время сутки", "утро добрый", "день добрый", "вечер добрый", "ночь добрый"]
farewells = ["до свидание", "до встреча", "все хороший", "все добрый", "все наилучший", "всего хороший", "всего добрый",
                 "всего наилучший", "весь хороший", "весь добрый", "весь наилучший"]
introductions = ["я", "я звать", "мой имя"]

morph = pymorphy2.MorphAnalyzer()

greeting_checks = []
farewell_checks = []
introduction_checks = []
names = []
companies = []

for _, df_str in df_data.iterrows():

    greeting_check = False
    farewell_check = False
    introduction_check = False
    name = ''
    company_name = ''

    if df_str['role'] != 'manager':
        greeting_checks.append(greeting_check)
        farewell_checks.append(farewell_check)
        introduction_checks.append(introduction_check)
        names.append(name)
        companies.append(company_name)
        continue
    
    text = df_str['text']

    norm_text = ''
    for word in word_tokenize(text):
        norm_text += morph.normal_forms(word)[0]
        norm_text += ' '

    for greeting in greetings:
        reg = r'\b' + greeting + r'\b'
        if re.search(reg, norm_text):
            greeting_check = True
            break
    greeting_checks.append(greeting_check)

    for farewell in farewells:
        reg = r'\b' + farewell + r'\b'
        if re.search(reg, norm_text):
            farewell_check = True
            break  
    farewell_checks.append(farewell_check) 

    for introduction in introductions:
        reg = r'\b' + introduction + r'\b'
        for match in re.finditer(reg, norm_text):
            end_pos = match.end()
            if end_pos < len(norm_text):
                text_after_intro = norm_text[end_pos:]
                for word in word_tokenize(text_after_intro):
                    if 'Name' in morph.parse(word)[0].tag:
                        name+= word
                        name+= ' '
                    else:
                        break
            if name != '':
                break
        
        if name != '':
            introduction_check = True

            # Пробовал для поиска компании разные NER (spacy, natasha, deeppavlov), 
            # но они все обучены, видимо, на данных, где текст в литературном формате.
            # Ни одна из моделей не находила нормально названия компаний.
            # Syntax parser тоже не распознавал нормально, так что решил остановиться на
            # простом и не точном варианте. Как находить названия компаний хорошо - не 
            # знаю

            
            comp_pos = norm_text.find("компания")
            if comp_pos != -1:
                end_pos = comp_pos + len("компания")
                company_name = word_tokenize(norm_text[end_pos:])[0]
        
            break
        
    introduction_checks.append(introduction_check)
    names.append(name)
    companies.append(company_name)

df_data.insert(4, "manager_greeting", greeting_checks)
df_data.insert(5, "manager_farewell", farewell_checks)
df_data.insert(6, "manager_introduction", introduction_checks)
df_data.insert(7, "name", names)   
df_data.insert(8, "company", companies) 

df_data.to_csv('processed_data.csv', index = False)

dlgs = []
demands = []

for dlg_id in df_data['dlg_id'].unique():

    df_diag_data = df_data.loc[np.bitwise_and(df_data['dlg_id'] == dlg_id, df_data['role'] == 'manager')]

    dlgs.append(dlg_id)
    if True in df_diag_data['manager_greeting'].unique() and True in df_diag_data['manager_farewell'].unique():
        demands.append(True)
    else:
        demands.append(False)

df_demands = pd.DataFrame({'dlg_id': dlgs, 'demand_check': demands})

df_demands.to_csv('demand_check.csv', index=False)
