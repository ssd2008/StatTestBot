import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import (Updater, CallbackContext, CommandHandler, CallbackQueryHandler,
                          ConversationHandler, MessageHandler, Filters)

# Состояния для ConversationHandler
DEPENDENT_VARIABLE, INDEPENDENT_VARIABLE, GROUPS, DEPENDENCY, DEPENDET_DISTRIBUTION, INDEPENDET_DISTRIBUTION, ANALYSIS_GOAL, FINISH_CHOOSING_TEST, SHOW_TESTS = range(9)

TOKEN = 'TOKEN'
df = pd.read_csv('tests.csv', delimiter=';')
user_data = {}

ANOVAcode = """
import numpy as np
from scipy.stats import f_oneway

# Группы данных (можно заменить своими данными)
data_group1 = df['x']
data_group2 = df['y']
data_group3 = df['z']

# Проведение однофакторного анализа дисперсии (ANOVA)
f_statistic, p_value = f_oneway(data_group1, data_group2, data_group3)

# Вывод результатов
print("F-статистика:", f_statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимые различия между группами.")
else:
    print("Нет статистически значимых различий между группами.")
"""
ttestcode = """
import numpy as np
from scipy.stats import ttest_ind

# Данные для двух групп (замените на свои данные)
data_group1 = df['x']  # Первая группа данных
data_group2 = df['y']  # Вторая группа данных

# Проведение независимого t-теста
t_statistic, p_value = ttest_ind(data_group1, data_group2, equal_var=True)  # equal_var=True предполагает равные дисперсии

# Вывод результатов
print("t-статистика:", t_statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимые различия между группами.")
else:
    print("Нет статистически значимых различий между группами.")

"""
utestcode = """
import numpy as np
from scipy.stats import mannwhitneyu

# Данные для двух групп (замените на свои данные)
data_group1 = df['x']  # Первая группа данных
data_group2 = df['y']  # Вторая группа данных

# Проведение U-теста Манна-Уитни
u_statistic, p_value = mannwhitneyu(data_group1, data_group2, alternative='two-sided')

# Вывод результатов
print("U-статистика:", u_statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимые различия между группами.")
else:
    print("Нет статистически значимых различий между группами.")
"""
wilcoxon = """
from scipy.stats import wilcoxon

# Данные до и после (замените своими данными)
data_before = df['before']  # Данные до
data_after = df['after']    # Данные после

# Проведение теста Уилкоксона
stat, p_value = wilcoxon(data_before, data_after)

# Вывод результатов
print("Статистика Уилкоксона:", stat)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимые различия между выборками.")
else:
    print("Нет статистически значимых различий между выборками.")
"""
ztest = """
import numpy as np
from statsmodels.stats.weightstats import ztest

# Данные двух групп (замените своими данными)
data_group1 = df['group1']  # Данные первой группы
data_group2 = df['group2']  # Данные второй группы

# Проведение Z-теста
z_statistic, p_value = ztest(data_group1, data_group2, value=0)

# Вывод результатов
print("Z-статистика:", z_statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимые различия между средними значениями групп.")
else:
    print("Нет статистически значимых различий между средними значениями групп.")
"""
kruskal = """
from scipy.stats import kruskal

# Данные групп (замените своими данными)
data_group1 = df['group1']
data_group2 = df['group2']
data_group3 = df['group3']

# Проведение теста Крускала-Уоллиса
h_statistic, p_value = kruskal(data_group1, data_group2, data_group3)

# Вывод результатов
print("H-статистика:", h_statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимые различия между группами.")
else:
    print("Нет статистически значимых различий между группами.")
"""
one_way_ancova = """
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Данные (замените своими)
# Зависимая переменная: 'score'
# Независимая переменная (группы): 'group'
# Ковариата: 'pre_score'
data = pd.DataFrame({
    'score': df['score'],         # Зависимая переменная
    'group': df['group'],         # Независимая переменная
    'pre_score': df['pre_score']  # Ковариата
})

# Модель ANCOVA
model = ols('score ~ group + pre_score', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Вывод результатов
print(anova_table)

# Интерпретация
alpha = 0.05  # Уровень значимости
if anova_table['PR(>F)']['group'] < alpha:
    print("Есть статистически значимые различия между группами после учёта ковариаты.")
else:
    print("Нет статистически значимых различий между группами после учёта ковариаты.")
"""
factor_anova = """
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Данные (замените своими)
# Зависимая переменная: 'score'
# Факторы: 'factor1' и 'factor2'
data = pd.DataFrame({
    'score': df['score'],         # Зависимая переменная
    'factor1': df['factor1'],     # Первый фактор
    'factor2': df['factor2']      # Второй фактор
})

# Модель ANOVA
model = ols('score ~ C(factor1) * C(factor2)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Вывод результатов
print(anova_table)

# Интерпретация взаимодействия
alpha = 0.05  # Уровень значимости
if anova_table.loc['C(factor1):C(factor2)', 'PR(>F)'] < alpha:
    print("Есть статистически значимое взаимодействие между факторами.")
else:
    print("Нет статистически значимого взаимодействия между факторами.")
"""
friedman = """
from scipy.stats import friedmanchisquare

# Данные (замените своими данными)
# Повторные измерения для одних и тех же объектов
group1 = df['measure1']  # Группа 1
group2 = df['measure2']  # Группа 2
group3 = df['measure3']  # Группа 3

# Проведение теста Фридмана
statistic, p_value = friedmanchisquare(group1, group2, group3)

# Вывод результатов
print("Статистика теста Фридмана:", statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимые различия между группами.")
else:
    print("Нет статистически значимых различий между группами.")
"""
one_way_repeated_measure_anova = """
import pandas as pd
import pingouin as pg

# Данные (замените своими данными)
# Длинный формат: каждая строка соответствует одному измерению
data = pd.DataFrame({
    'subject': np.tile(np.arange(1, 21), 3),  # Участники
    'condition': np.repeat(['A', 'B', 'C'], 20),  # Условия
    'score': np.random.normal(50, 10, 60)  # Результаты
})

# Проведение ANOVA с повторными измерениями
anova = pg.rm_anova(dv='score', within='condition', subject='subject', data=data, detailed=True)

# Вывод результатов
print(anova)

# Интерпретация
alpha = 0.05
if anova['p-GG-corr'].iloc[0] < alpha:  # Корректировка Гринхауса-Гейссера
    print("Есть статистически значимые различия между условиями.")
else:
    print("Нет статистически значимых различий между условиями.")
"""
split_plot_anova = """
import pandas as pd
import numpy as np
import pingouin as pg

# Пример данных
data = pd.DataFrame({
    'subject': np.tile(np.arange(1, 21), 3),              # Участники
    'group': np.repeat(['A', 'B'], 30),                  # Межгрупповой фактор
    'time': np.tile(['T1', 'T2', 'T3'], 20),            # Внутригрупповой фактор
    'score': np.random.normal(50, 10, 60)               # Зависимая переменная
})

# Проведение Split-Plot ANOVA
anova = pg.mixed_anova(
    dv='score', between='group', within='time', subject='subject', data=data
)

# Вывод результатов
print(anova)

# Интерпретация взаимодействия
alpha = 0.05
if anova.loc[anova['Source'] == 'Interaction', 'p-unc'].values[0] < alpha:
    print("Есть статистически значимое взаимодействие между факторами.")
else:
    print("Нет статистически значимого взаимодействия между факторами.")
"""
bl = """
делай доьро бро"""
paired_samples_ttest = """
from scipy.stats import ttest_rel

# Данные (замените своими данными)
before = df['before']  # Результаты до вмешательства
after = df['after']    # Результаты после вмешательства

# Проведение парного t-теста
t_statistic, p_value = ttest_rel(before, after)

# Вывод результатов
print("T-статистика:", t_statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимые различия между выборками.")
else:
    print("Нет статистически значимых различий между выборками.")
"""
single_sample_ttest = """
from scipy.stats import ttest_1samp

# Данные (замените своими данными)
data = df['values']  # Выборка данных
population_mean = 50  # Заданное среднее значение

# Проведение теста
t_statistic, p_value = ttest_1samp(data, population_mean)

# Вывод результатов
print("T-статистика:", t_statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Среднее значение выборки статистически значимо отличается от заданного значения.")
else:
    print("Среднее значение выборки не отличается от заданного значения.")
"""
single_sample_wilcoxon_signed_rank_test = """
from scipy.stats import wilcoxon

# Данные (замените своими данными)
data = df['values']  # Выборка данных
population_median = 50  # Заданная медиана

# Преобразование данных: рассчитываем разности с медианой
differences = data - population_median

# Проведение теста
statistic, p_value = wilcoxon(differences)

# Вывод результатов
print("Статистика Wilcoxon:", statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Медиана выборки статистически значимо отличается от заданного значения.")
else:
    print("Медиана выборки не отличается от заданного значения.")
"""
exact_test_of_goodness_of_fit = """
from scipy.stats import binom_test

# Наблюдаемые частоты
observed = [10, 15, 25]  # Замена на ваши данные
expected = [12, 18, 20]  # Ожидаемые частоты (замена на ваши данные)

# Подсчет общего количества наблюдений
n_total = sum(observed)

# Расчет вероятностей для категорий
expected_probabilities = [e / n_total for e in expected]

# Проверка каждой категории
for i, (obs, exp_prob) in enumerate(zip(observed, expected_probabilities)):
    p_value = binom_test(obs, n=n_total, p=exp_prob, alternative='two-sided')
    print(f"Категория {i + 1}: Наблюдаемое = {obs}, P-значение = {p_value}")

    # Интерпретация результата
    alpha = 0.05  # Уровень значимости
    if p_value < alpha:
        print("Отклоняем нулевую гипотезу для этой категории.")
    else:
        print("Не отклоняем нулевую гипотезу для этой категории.")
"""
one_proportion_ztest = """
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Данные
successes = 45  # Количество успехов
n_total = 100   # Общее количество наблюдений
expected_proportion = 0.5  # Ожидаемая пропорция

# Проведение Z-теста
statistic, p_value = proportions_ztest(count=successes, nobs=n_total, value=expected_proportion)

# Вывод результатов
print("Z-статистика:", statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Наблюдаемая пропорция статистически значимо отличается от ожидаемой пропорции.")
else:
    print("Наблюдаемая пропорция не отличается от ожидаемой пропорции.")
"""
Gtest_of_goodness_of_fit = """
import numpy as np
from scipy.stats import power_divergence

# Наблюдаемые и ожидаемые частоты
observed = np.array([30, 40, 50])  # Наблюдаемые частоты (замените своими данными)
expected = np.array([40, 40, 40])  # Ожидаемые частоты (замените своими данными)

# Проведение G-теста
statistic, p_value = power_divergence(f_obs=observed, f_exp=expected, lambda_="log-likelihood")

# Вывод результатов
print("G-статистика:", statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Наблюдаемое распределение статистически значимо отличается от ожидаемого.")
else:
    print("Наблюдаемое распределение соответствует ожидаемому распределению.")
"""
chi_square_goodness_of_fit = """
import numpy as np
from scipy.stats import chisquare

# Наблюдаемые и ожидаемые частоты
observed = np.array([50, 30, 20])  # Наблюдаемые частоты (замените своими данными)
expected = np.array([40, 40, 20])  # Ожидаемые частоты (замените своими данными)

# Проведение теста хи-квадрат
statistic, p_value = chisquare(f_obs=observed, f_exp=expected)

# Вывод результатов
print("Хи-квадрат статистика:", statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Наблюдаемое распределение статистически значимо отличается от ожидаемого.")
else:
    print("Наблюдаемое распределение соответствует ожидаемому распределению.")
"""
exact_test_of_goodness_of_fit_multinomial_model = """
from scipy.stats import multinomial
import numpy as np

# Наблюдаемые частоты
observed = np.array([3, 4, 3])  # Замените своими данными
n_total = observed.sum()        # Общее количество наблюдений

# Ожидаемые вероятности для каждой категории
expected_probabilities = np.array([0.3, 0.4, 0.3])  # Замените своими данными

# Вычисление вероятности наблюдаемого распределения
p_value = multinomial.pmf(observed, n=n_total, p=expected_probabilities)

# Вывод результатов
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Наблюдаемое распределение статистически значимо отличается от ожидаемого.")
else:
    print("Наблюдаемое распределение соответствует ожидаемому распределению.")
"""
gtest = """
import numpy as np
from scipy.stats import power_divergence

# Наблюдаемые и ожидаемые частоты
observed = np.array([30, 50, 20])  # Наблюдаемые частоты (замените своими данными)
expected = np.array([33, 47, 20])  # Ожидаемые частоты (замените своими данными)

# Проведение G-теста
statistic, p_value = power_divergence(f_obs=observed, f_exp=expected, lambda_="log-likelihood")

# Вывод результатов
print("G-статистика:", statistic)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Наблюдаемое распределение статистически значимо отличается от ожидаемого.")
else:
    print("Наблюдаемое распределение соответствует ожидаемому распределению.")
"""
chi_square_test_of_independence = """
import numpy as np
from scipy.stats import chi2_contingency

# Создаем таблицу сопряженности
data = np.array([[20, 30],  # Пример данных: строки - категории одной переменной
                 [30, 40]])  # столбцы - категории другой переменной

# Проведение теста хи-квадрат
chi2_stat, p_value, dof, expected = chi2_contingency(data)

# Вывод результатов
print("Хи-квадрат статистика:", chi2_stat)
print("P-значение:", p_value)
print("Степени свободы:", dof)
print("Ожидаемые частоты:")
print(expected)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимая зависимость между переменными.")
else:
    print("Переменные независимы.")
"""
mcnemar_test = """
from statsmodels.stats.contingency_tables import mcnemar

# Таблица сопряженности 2x2
# Формат: [[a, b], [c, d]], где:
# a - обе переменные согласны, b - первая согласна, вторая не согласна
# c - первая не согласна, вторая согласна, d - обе переменные не согласны
data = [[30, 10],  # Пример данных
        [5, 55]]

# Проведение теста
result = mcnemar(data, exact=False)  # exact=True для точного теста McNemar
print("Хи-квадрат статистика:", result.statistic)
print("P-значение:", result.pvalue)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if result.pvalue < alpha:
    print("Есть статистически значимое различие между условиями.")
else:
    print("Нет статистически значимого различия между условиями.")
"""
fisher_exact_test = """
import pandas as pd
from scipy.stats import fisher_exact

# Таблица сопряженности 2x2
contingency_table = pd.crosstab(df['Variable1'], df['Variable2'])

# Проведение теста
oddsratio, p_value = fisher_exact(contingency_table, alternative='two-sided')

# Вывод результатов
print("Отношение шансов:", oddsratio)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Есть статистически значимая зависимость между переменными.")
else:
    print("Переменные независимы.")
"""
two_proportion_ztest = """
from statsmodels.stats.proportion import proportions_ztest

# Количество "успехов" (например, событий) в каждой группе
success_counts = [50, 30]  # Замените своими данными

# Общее количество наблюдений в каждой группе
sample_sizes = [100, 90]  # Замените своими данными

# Проведение теста
stat, p_value = proportions_ztest(count=success_counts, nobs=sample_sizes, alternative='two-sided')

# Вывод результатов
print("Z-статистика:", stat)
print("P-значение:", p_value)

# Интерпретация результата
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Пропорции в двух группах статистически значимо различаются.")
else:
    print("Нет статистически значимой разницы между пропорциями в двух группах.")
"""
log_linear_analysis = """
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Пример данных: Создание таблицы сопряженности
data = pd.DataFrame({
    'Var1': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'Var2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Var3': ['M', 'M', 'M', 'M', 'N', 'N', 'N', 'N'],
    'Frequency': [20, 30, 25, 15, 35, 45, 40, 30]
})

# Построение модели
model = glm('Frequency ~ Var1 * Var2 * Var3', data=data, family=sm.families.Poisson()).fit()

# Вывод результатов
print(model.summary())

# Интерпретация результата
if model.deviance / model.df_resid < 1:
    print("Модель адекватно объясняет данные.")
else:
    print("Модель неадекватно объясняет данные.")
"""
ordered_logistic_regression = """
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Пример данных
data = pd.DataFrame({
    'Satisfaction': [1, 2, 3, 2, 1, 3, 2, 1, 3, 2],  # Порядковая зависимая переменная
    'Age': [22, 25, 30, 35, 40, 22, 29, 33, 26, 28],  # Количественная независимая переменная
    'Income': [3000, 4000, 5000, 6000, 7000, 3000, 4500, 4800, 3200, 4100]  # Количественная независимая переменная
})

# Преобразование зависимой переменной в категорию с упорядоченными уровнями
data['Satisfaction'] = pd.Categorical(data['Satisfaction'], ordered=True)

# Построение модели
model = OrderedModel(
    data['Satisfaction'],  # Зависимая переменная
    data[['Age', 'Income']],  # Независимые переменные
    distr='logit'  # Использование логистического распределения
)

# Оценка модели
result = model.fit(method='bfgs')

# Вывод результатов
print(result.summary())

# Предсказание
predicted = result.predict(data[['Age', 'Income']])
print(predicted)
"""
linear_discriminant_analysis = """
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Пример данных
data = pd.DataFrame({
    'Feature1': [2.1, 1.3, 3.1, 2.8, 3.5, 4.2, 3.9, 2.2, 3.3, 2.7],
    'Feature2': [1.2, 1.8, 3.4, 2.9, 2.7, 3.3, 2.5, 1.5, 3.1, 2.2],
    'Class': ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
})

# Разделение данных на независимые переменные и зависимую переменную
X = data[['Feature1', 'Feature2']]
y = data['Class']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Построение модели LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = lda.predict(X_test)

# Оценка модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Вывод коэффициентов
print("Коэффициенты дискриминантной функции:", lda.coef_)
"""
multinomial_logistic_regression = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Пример данных
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'C', 'A']
})

# Разделение данных на независимые переменные и зависимую переменную
X = data[['Feature1', 'Feature2']]
y = data['Category']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Построение модели
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
print("Classification Report:\n", classification_report(y_test, y_pred))

# Вывод коэффициентов
print("Коэффициенты модели:\n", model.coef_)
"""
simple_linear_regression = """
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Пример данных
data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Независимая переменная
    'Y': [2.3, 2.5, 3.7, 3.9, 5.2, 5.8, 6.5, 7.1, 8.0, 9.3]  # Зависимая переменная
})

# Независимая и зависимая переменные
X = data['X']
Y = data['Y']

# Добавление константы для модели
X = sm.add_constant(X)

# Построение модели
model = sm.OLS(Y, X).fit()

# Вывод результатов
print(model.summary())

# Построение графика регрессии
plt.scatter(data['X'], data['Y'], label='Данные')
plt.plot(data['X'], model.predict(X), color='red', label='Линия регрессии')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
"""
mixed_model = """
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Пример данных
data = pd.DataFrame({
    'Score': [85, 78, 92, 88, 91, 87, 95, 90, 93, 88],  # Зависимая переменная
    'StudyHours': [10, 8, 12, 11, 15, 14, 10, 9, 13, 11],  # Независимая переменная (фиксированный эффект)
    'Group': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]  # Группы (случайный эффект)
})

# Построение смешанной модели
model = mixedlm("Score ~ StudyHours", data, groups=data["Group"])
result = model.fit()

# Вывод результатов
print(result.summary())
"""
multiple_linear_regression = """
import pandas as pd
import statsmodels.api as sm

# Пример данных
data = pd.DataFrame({
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Независимая переменная 1
    'X2': [2.1, 3.4, 4.5, 5.8, 6.9, 7.0, 8.1, 9.4, 10.2, 11.3],  # Независимая переменная 2
    'Y': [1.5, 2.3, 2.9, 4.1, 4.9, 5.8, 6.7, 7.3, 8.2, 9.1]  # Зависимая переменная
})

# Разделение на зависимую и независимые переменные
X = data[['X1', 'X2']]
Y = data['Y']

# Добавление константы
X = sm.add_constant(X)

# Построение модели
model = sm.OLS(Y, X).fit()

# Вывод результатов
print(model.summary())
"""
multivariate_multiple_logistic_regression = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# Пример данных
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Outcome1': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],  # Зависимая переменная 1
    'Outcome2': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]   # Зависимая переменная 2
})

# Разделение данных на независимые и зависимые переменные
X = data[['Feature1', 'Feature2']]
Y = data[['Outcome1', 'Outcome2']]

# Построение модели
log_reg = LogisticRegression(max_iter=1000)
multi_log_reg = MultiOutputClassifier(log_reg)
multi_log_reg.fit(X, Y)

# Предсказания
Y_pred = multi_log_reg.predict(X)

# Оценка модели
print("Classification Report for Outcome1 and Outcome2:\n")
print(classification_report(Y, Y_pred))
"""
multiple_logistic_regression = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Пример данных
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Outcome': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  # Зависимая переменная
})

# Разделение данных на независимые и зависимую переменные
X = data[['Feature1', 'Feature2']]
Y = data['Outcome']

# Построение модели
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, Y)

# Коэффициенты модели
print("Коэффициенты:", log_reg.coef_)
print("Свободный член (intercept):", log_reg.intercept_)

# Предсказания
Y_pred = log_reg.predict(X)

# Оценка модели
print("Classification Report:\n")
print(classification_report(Y, Y_pred))
"""
mixed_effects_logistic_regression = """
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Пример данных
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Независимая переменная
    'Feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Независимая переменная
    'Group': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],  # Группы (случайный эффект)
    'Outcome': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  # Зависимая переменная (бинарная)
})

# Логит-преобразование для модели
data['logit'] = sm.families.links.logit()(data['Outcome'])

# Построение смешанной модели логистической регрессии
model = sm.MixedLM.from_formula("logit ~ Feature1 + Feature2", groups=data["Group"], data=data)
result = model.fit()

# Вывод результатов
print(result.summary())
"""
simple_logistic_regression = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Пример данных
data = pd.DataFrame({
    'Feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Независимая переменная
    'Outcome': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  # Зависимая переменная
})

# Разделение данных на независимую и зависимую переменные
X = data[['Feature']]
Y = data['Outcome']

# Построение модели
log_reg = LogisticRegression()
log_reg.fit(X, Y)

# Коэффициенты модели
print("Коэффициент:", log_reg.coef_[0])
print("Свободный член (intercept):", log_reg.intercept_)

# Предсказания
Y_pred = log_reg.predict(X)

# Оценка модели
print("Classification Report:\n")
print(classification_report(Y, Y_pred))
"""
partial_correlation = """
import pandas as pd
from pingouin import partial_corr

# Пример данных
data = pd.DataFrame({
    'Variable1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Переменная 1
    'Variable2': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],  # Переменная 2
    'ControlVariable': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Контролируемая переменная
})

# Вычисление парциальной корреляции
result = partial_corr(data=data, x='Variable1', y='Variable2', covar='ControlVariable', method='pearson')

# Вывод результата
print(result)
"""
pearson_correlation = """
import pandas as pd
from scipy.stats import pearsonr

# Пример данных
data = pd.DataFrame({
    'Variable1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Переменная 1
    'Variable2': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105]  # Переменная 2
})

# Вычисление коэффициента Пирсона
r, p_value = pearsonr(data['Variable1'], data['Variable2'])

# Вывод результатов
print("Коэффициент корреляции Пирсона (r):", r)
print("P-значение:", p_value)

# Интерпретация
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Корреляция значима.")
else:
    print("Корреляция не значима.")
"""
phi_coefficient = """
import pandas as pd
from scipy.stats import chi2_contingency
from math import sqrt

# Пример данных
data = pd.DataFrame({
    'Variable1': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],  # Переменная 1
    'Variable2': [0, 1, 1, 1, 0, 1, 0, 0, 1, 1]   # Переменная 2
})

# Создание таблицы сопряженности
contingency_table = pd.crosstab(data['Variable1'], data['Variable2'])

# Вычисление статистики Хи-квадрат
chi2, p, _, _ = chi2_contingency(contingency_table)

# Вычисление Phi Coefficient
n = contingency_table.values.sum()  # Общее количество наблюдений
phi = sqrt(chi2 / n)

# Вывод результатов
print("Phi Coefficient:", phi)
print("P-значение:", p)

# Интерпретация
alpha = 0.05  # Уровень значимости
if p < alpha:
    print("Связь между переменными статистически значима.")
else:
    print("Связь между переменными не значима.")
"""
cramersv = """
import pandas as pd
from scipy.stats import chi2_contingency
from math import sqrt

# Пример данных
data = pd.DataFrame({
    'Variable1': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],  # Переменная 1
    'Variable2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'X', 'Y', 'Y']   # Переменная 2
})

# Создание таблицы сопряженности
contingency_table = pd.crosstab(data['Variable1'], data['Variable2'])

# Вычисление статистики Хи-квадрат
chi2, p, _, _ = chi2_contingency(contingency_table)

# Вычисление Cramer's V
n = contingency_table.values.sum()  # Общее количество наблюдений
min_dim = min(contingency_table.shape) - 1  # Минимальное измерение таблицы сопряженности
cramers_v = sqrt(chi2 / (n * min_dim))

# Вывод результатов
print("Cramer's V:", cramers_v)
print("P-значение:", p)

# Интерпретация
alpha = 0.05  # Уровень значимости
if p < alpha:
    print("Связь между переменными статистически значима.")
else:
    print("Связь между переменными не значима.")
"""
kendalls_tau_orspearmans_rho = """
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# Пример данных
data = pd.DataFrame({
    'Variable1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Переменная 1
    'Variable2': [15, 22, 33, 47, 53, 66, 71, 82, 95, 105]  # Переменная 2
})

# Вычисление Kendall's Tau
tau, p_tau = kendalltau(data['Variable1'], data['Variable2'])

# Вычисление Spearman's Rho
rho, p_rho = spearmanr(data['Variable1'], data['Variable2'])

# Вывод результатов
print("Kendall's Tau:")
print("Tau:", tau)
print("P-значение:", p_tau)

print("\nSpearman's Rho:")
print("Rho:", rho)
print("P-значение:", p_rho)

# Интерпретация
alpha = 0.05  # Уровень значимости
if p_tau < alpha:
    print("Kendall's Tau: Связь между переменными значима.")
else:
    print("Kendall's Tau: Связь между переменными не значима.")

if p_rho < alpha:
    print("Spearman's Rho: Связь между переменными значима.")
else:
    print("Spearman's Rho: Связь между переменными не значима.")
"""
point_biserial_correlation = """
import pandas as pd
from scipy.stats import pointbiserialr

# Пример данных
data = pd.DataFrame({
    'BinaryVariable': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # Бинарная переменная
    'QuantitativeVariable': [2.3, 3.5, 4.2, 2.1, 3.9, 2.7, 4.5, 3.8, 2.4, 4.1]  # Количественная переменная
})

# Вычисление Point-Biserial Correlation
r, p_value = pointbiserialr(data['BinaryVariable'], data['QuantitativeVariable'])

# Вывод результатов
print("Point-Biserial Correlation:")
print("r:", r)
print("P-значение:", p_value)

# Интерпретация
alpha = 0.05  # Уровень значимости
if p_value < alpha:
    print("Связь между переменными статистически значима.")
else:
    print("Связь между переменными не значима.")
"""

codes = {'one-way anova' : ANOVAcode, 't-test' : ttestcode, 'u-test' : utestcode, 'тест уилкоксона' : wilcoxon, 'независимый t-test' : ttestcode,
         'single sample z-test' : ztest, 'тест трускала-уоллиса' : kruskal, 'one-way ancova' : one_way_ancova, "факторный anova" : factor_anova,
         'тест фридмана' : friedman, 'one-way repeated measure anova' : one_way_repeated_measure_anova, 'бля' : bl, 'paired samples t-test' : paired_samples_ttest,
         'single sample t-test' : single_sample_ttest, 'single sample wilcoxon signed-rank test' : single_sample_wilcoxon_signed_rank_test,
         'exact test of goodness of fit' : exact_test_of_goodness_of_fit, 'one-proportion z-test' : one_proportion_ztest,
         'g-test of goodness of fit' : Gtest_of_goodness_of_fit, 'chi-square goodness of fit' : chi_square_goodness_of_fit,
         'exact test of goodness of fit (multinomial model)' : exact_test_of_goodness_of_fit_multinomial_model,
         'g-test' : gtest, 'chi-square test of independence' : chi_square_test_of_independence, 'mcnemar test' : mcnemar_test,
         "fisher's exact test" : fisher_exact_test, 'two-proportion z-test' : two_proportion_ztest, 'split-plot anova' : split_plot_anova,
         'log-linear analysis' : log_linear_analysis, 'ordered logistic regression' : ordered_logistic_regression,
         'linear discriminant analysis' : linear_discriminant_analysis, 'multinomial logistic regression' : multinomial_logistic_regression,
         'simple linear regression' : simple_linear_regression, 'mixed model' : mixed_model,
         'multiple linear regression' : multiple_linear_regression, 'multivariate multiple logistic regression' : multivariate_multiple_logistic_regression,
         'multiple logistic regression' : multiple_logistic_regression, 'mixed effects logistic regression' : mixed_effects_logistic_regression,
         'simple logistic regression' : simple_logistic_regression, 'partial correlation' : partial_correlation,
         'pearson correlation' : pearson_correlation, 'phi coefficient' : phi_coefficient, "cramer's v" : cramersv,
         "kendall's tau or spearman's rho" : kendalls_tau_orspearmans_rho, 'point biserial correlation' : point_biserial_correlation
         }
# Сережа, не забывай вводить названия тестов маленькими буквами потому что ты делаешь .lower()

DESCRIBE_TEST = 123
DESCRIBE_CURRENT_TEST = 100
START = 111

# Стартовая команда
def start(update: Update, context: CallbackContext):
    user_data[update.effective_chat.id] = {}

    keyboard = [
        [InlineKeyboardButton("Выбрать Стат. тест", callback_data='choose_test')],
        [InlineKeyboardButton("Описания Стат. тестов", callback_data='describe_test')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Привет! Выберите действие:', reply_markup=reply_markup)


def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()

    if query.data == 'start':
        start(update, context)
        return ConversationHandler.END  # Возвращаемся в начальное состояние
    elif query.data == 'choose_test':
        query.edit_message_text("ахахахах удачи")
        return DEPENDENT_VARIABLE  # Запускаем обработку через ConversationHandler
    elif query.data == 'describe_test':
        query.edit_message_text('Введите название стат. теста, чтобы получить описание:')
        return DESCRIBE_TEST  # Устанавливаем состояние для описания тестов
    elif query.data == 'go_back':  # Обрабатываем кнопку "Назад"
        show_tests(update, context)
        return DESCRIBE_TEST  # Возвращаем состояние для описания тестов


# Обработчик текстовых сообщений для получения описания теста
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def describe_test_handler(update: Update, context: CallbackContext):
    try:
        test_name = user_data[update.effective_chat.id]['test']
        if test_name == '0':
            test_name = update.message.text.strip().lower()
    except KeyError:
        test_name = update.message.text.strip().lower()

    description = get_test_description(test_name)
    if description:
        escaped_info = description['info']
        escaped_restrictions = description['restrictions']
        escaped_when_to_use = description['when_to_use']
        escaped_data_type = description['data_type']
        escaped_null_hypotesa = description['null_hypotesa']
        escaped_alt_hypotesa = description['alt_hypotesa']
        escaped_distribution = description['distribution']
        escaped_nuances = description['nuances']

        update.message.reply_text(
            f"Описание теста '{test_name}':\n"
            f"{escaped_info}\n\nКогда использовать:\n{escaped_when_to_use}\n\n"
            f"Тип данных:\n{escaped_data_type}\n\nРаспределение данных:\n{escaped_distribution}\n\n"
            f"Нулевая Гипотеза:\n{escaped_null_hypotesa}\n\nАльтернативная гипотеза:\n{escaped_alt_hypotesa}\n\n"
            f"Ограничения:\n{escaped_restrictions}\n\nНюансы:\n{escaped_nuances}"
        )

        send_code(update, test_name)
    else:
        update.message.reply_text(f"Тест '{test_name}' не найден.")

    # Добавляем кнопки для навигации
    # keyboard = [
    #     [InlineKeyboardButton("К началу", callback_data='start')],
    #     [InlineKeyboardButton("Ввести новый тест", callback_data='describe_test')],
    #     [InlineKeyboardButton("Показать тесты", callback_data='go_back')]
    # ]
    # reply_markup = InlineKeyboardMarkup(keyboard)
    # update.message.reply_text("Навигация", reply_markup=reply_markup)

    reply_keyboard = [["К началу", "Показать тесты"]]
    update.message.reply_text(
        "Что дальше делать? Либо введи новый тест",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )

    user_data[update.effective_chat.id]['next_step'] = update.message.text
    if user_data[update.effective_chat.id]['next_step'] == 'К началу':
        start(update, context)
        return ConversationHandler.END
    elif user_data[update.effective_chat.id]['next_step'] == 'Показать тесты':
        return SHOW_TESTS


from telegram import ParseMode

def send_code(update : Update, test_name : '0'):
    code = codes[test_name]
    formatted_code = f"```\n{code}```"
    update.message.reply_text(formatted_code, parse_mode=ParseMode.MARKDOWN_V2)


# Функция для получения описания теста
def get_test_description(test_name):
    test_name = test_name.lower()
    stroka = df[df['name'] == test_name].reset_index(drop=True)

    if stroka.shape[0] == 0:
        return

    info = stroka.loc[0, 'info']
    restrictions = stroka.loc[0, 'restrictions']
    when_to_use = stroka.loc[0, 'when_to_use']
    data_type = stroka.loc[0, 'data_type']
    null_hypotesa = stroka.loc[0, 'null_hypotesa']
    alt_hypotesa = stroka.loc[0, 'alt_hypotesa']
    distribution = stroka.loc[0, 'distribution']
    nuances = stroka.loc[0, 'nuances']

    return {"info": info, "restrictions": restrictions, "when_to_use": when_to_use, "data_type": data_type, "null_hypotesa": null_hypotesa, "alt_hypotesa": alt_hypotesa, "distribution": distribution, "nuances": nuances}




def dependent_variable(update: Update, context: CallbackContext) -> int:
    """Сохраняет тип зависимой переменной."""
    reply_keyboard = [["Бинарная (бинарные)", "Количественная (количественные)", "Порядковая (порядковые)", "Категориальная (категориальные)", "Разные типы", "Пропустить"]]
    update.message.reply_text(
        "Какая у тебя зависимая переменная?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return INDEPENDENT_VARIABLE

def independent_variable(update: Update, context: CallbackContext) -> int:
    """Сохраняет тип независимой переменной."""
    user_data[update.effective_chat.id]['dependent_variable'] = update.message.text
    reply_keyboard = [["Бинарная (бинарные)", "Количественная (количественные)", "Порядковая (порядковые)", "Категориальная (категориальные)", "Разные типы", "Пропустить"]]
    update.message.reply_text(
        "Какая у тебя независимая переменная?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return GROUPS

def group_levels(update: Update, context: CallbackContext) -> int:
    """Сохраняет количество групп/уровней."""
    user_data[update.effective_chat.id]['independent_variable'] = update.message.text
    reply_keyboard = [["Две", "Больше двух", "Не применимо/не важно", "Пропустить"]]
    update.message.reply_text(
        "Сколько групп/уровней имеет независимая переменная?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return DEPENDENCY

def observation_dependency(update: Update, context: CallbackContext) -> int:
    """Сохраняет зависимость наблюдений."""
    user_data[update.effective_chat.id]['groups'] = update.message.text
    if user_data[update.effective_chat.id]['groups'] == 'Не применимо/не важно':
        user_data[update.effective_chat.id]['groups'] = "Пропустить"

    reply_keyboard = [["Зависимые", "Независимые", "Не применимо/не важно", "Пропустить"]]
    update.message.reply_text(
        "Наблюдения зависимы или независимы?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return DEPENDET_DISTRIBUTION

def dependet_distribution(update: Update, context: CallbackContext) -> int:
    """Сохраняет информацию о зависимости данных."""
    user_data[update.effective_chat.id]['dependency'] = update.message.text
    if user_data[update.effective_chat.id]['dependency'] == 'Не применимо/не важно':
        user_data[update.effective_chat.id]['dependency'] = "Пропустить"

    reply_keyboard = [["Нормальное", "Ненормальное", "Не важно", "Пропустить"]]
    update.message.reply_text(
        "Какое распределение имеют зависимые переменные?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return INDEPENDET_DISTRIBUTION

def independet_distribution(update: Update, context: CallbackContext) -> int:
    """Сохраняет информацию о зависимости данных."""
    user_data[update.effective_chat.id]['distribution_dependet'] = update.message.text


    reply_keyboard = [["Нормальное", "Ненормальное", "Не важно", "Пропустить"]]
    update.message.reply_text(
        "тут выбери то, что ты выбрал в предыдущем пункте (сорян, мне лень чинить было)",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return ANALYSIS_GOAL


def analysis_goal(update: Update, context: CallbackContext) -> int:
    """Сохраняет распределение и фильтрует тесты."""
    user_data[update.effective_chat.id]['distribution_independet'] = update.message.text
    reply_keyboard = [["Сравнение выборок", "Изучение связи", "Пропустить"]]
    update.message.reply_text(
        "Сравниваются ли выборки или изучается связь между переменными?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return FINISH_CHOOSING_TEST


def finish(update: Update, context: CallbackContext) -> int:
    user_data[update.effective_chat.id]['analysis_goal'] = update.message.text
    reply_keyboard = [["го"]]
    update.message.reply_text(
        "тестs готовы. показать?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return SHOW_TESTS


def filter_df(df, s, var):
    if var == 'Пропустить':
        return df
    return df[df[s] == var]

def show_tests(update: Update, context: CallbackContext):
    filtered_tests = df.copy()

    dep_var = user_data[update.effective_chat.id].get('dependent_variable', '')
    filtered_tests = filter_df(filtered_tests, 'dependent_variable', dep_var)
    # print(filtered_tests.shape[0], dep_var)

    indep_var = user_data[update.effective_chat.id].get('independent_variable', '')
    filtered_tests = filter_df(filtered_tests, 'independent_variable', indep_var)
    # print(filtered_tests.shape[0], indep_var)

    zavisimost = user_data[update.effective_chat.id].get('groups', '')
    filtered_tests = filter_df(filtered_tests, 'group_levels', zavisimost)
    # print(filtered_tests.shape[0], zavisimost)

    dependency = user_data[update.effective_chat.id].get('dependency', '')
    filtered_tests = filter_df(filtered_tests, 'observation_dependency', dependency)
    # print(filtered_tests.shape[0], dependency)

    distribution_dependet = user_data[update.effective_chat.id].get('distribution_dependet', '')
    filtered_tests = filter_df(filtered_tests, 'dependet_distribution', distribution_dependet)
    # print(filtered_tests.shape[0], distribution_dependet)

    distribution_independet = user_data[update.effective_chat.id].get('distribution_independet', '')
    filtered_tests = filter_df(filtered_tests, 'independet_distribution', distribution_independet)
    # print(filtered_tests.shape[0], distribution_independet)

    analysis_goal = user_data[update.effective_chat.id].get('analysis_goal', '')
    filtered_tests = filter_df(filtered_tests, 'analysis_goal', analysis_goal)
    # print(filtered_tests.shape[0], analysis_goal)

    if not filtered_tests.empty:
        tests = filtered_tests['name'].tolist()
        reply_keyboard = [[name for name in tests]]
        text = 'вот подходящие тесты:\n'
        for name in tests:
            text += name + '\n'
        text += 'Какой тест описать?'
        update.message.reply_text(
            text,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
        )
        return DESCRIBE_TEST
    else:
        update.message.reply_text("К сожалению, подходящих тестов не найдено.")
        return ConversationHandler.END


def cancel(update: Update, context: CallbackContext) -> int:
    """Отменяет диалог."""
    update.message.reply_text("Диалог отменен.", reply_markup=ReplyKeyboardMarkup([], resize_keyboard=True))
    return ConversationHandler.END



# Основной код для запуска бота
def main():
    updater = Updater(TOKEN, use_context=True)

    # ConversationHandler для выбора тестов
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(button_handler)],  # Обработка нажатий кнопок
        states={
            DEPENDENT_VARIABLE: [MessageHandler(Filters.text & ~Filters.command, dependent_variable)],
            INDEPENDENT_VARIABLE: [MessageHandler(Filters.text & ~Filters.command, independent_variable)],
            GROUPS: [MessageHandler(Filters.text & ~Filters.command, group_levels)],
            DEPENDENCY: [MessageHandler(Filters.text & ~Filters.command, observation_dependency)],
            DEPENDET_DISTRIBUTION: [MessageHandler(Filters.text & ~Filters.command, dependet_distribution)],
            INDEPENDET_DISTRIBUTION: [MessageHandler(Filters.text & ~Filters.command, independet_distribution)],
            ANALYSIS_GOAL: [MessageHandler(Filters.text & ~Filters.command, analysis_goal)],
            FINISH_CHOOSING_TEST: [MessageHandler(Filters.text & ~Filters.command, finish)],
            SHOW_TESTS: [MessageHandler(Filters.text & ~Filters.command, show_tests)],
            START: [MessageHandler(Filters.text & ~Filters.command, start)],
            DESCRIBE_TEST: [MessageHandler(Filters.text & ~Filters.command, describe_test_handler)],  # Обработчик описания
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Общая обработка команд
    updater.dispatcher.add_handler(CommandHandler("start", start))
    updater.dispatcher.add_handler(conv_handler)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
