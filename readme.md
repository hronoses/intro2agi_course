# Матеріали до курсу "Вступ до штучного загального інтелекту"

Цей репозиторій містить матеріали до курсу "Вступ до загального штучного інтелекту": лекційні презентації, практики та шаблони звітів для практичних завдань.

## Структура репозиторію:
- `lectures/` — презентації та матеріали до лекцій
- `practice/` — практичні завдання та шаблони звітів
- `problem_set/` — задачі для самостійного розв'язання (не обов'язкові)

для здачі домашніх завдань використовуйте шаблони звітів у папках до кожної практики (`practice_1/homework/`, `practice_2/homework/` тощо). Вони містять інструкції та структуру для оформлення  звіту (`report_template.qmd`).

## Початок роботи

``` bash
git clone https://github.com/hronoses/intro2agi_course
cd intro2agi_course
git checkout -b your_branch_name
pip install -r requirements.txt
```

Замініть `your_branch_name` на назву своєї гілки. Для оновлення матеріалів:

``` bash
git pull origin main
```

## Перегляд презентацій

Готовий HTML відкривається просто у браузері:

``` bash
xdg-open lectures/lecture_4/presentation.html   # Linux
open lectures/lecture_4/presentation.html        # macOS
start lectures/lecture_4/presentation.html       # Windows
```


## Домашні завдання

### Запустіть стартовий код та ознайомтесь із середовищем

``` bash
python practices/practice_1/1_binary_discrete.py
```

### Переглянути умову завдання

Pdf версія умови завдання знаходиться за адресою `practices/practice_1/homework/instructions.pdf`.

### Заповнити та здати звіт

1.  Скопіюйте шаблон під своє прізвище:

    ``` bash
    cp practices/practice_1/homework/report_template.qmd practices/practice_1/homework/report_Прізвище.qmd
    ```

2.  Заповніть відповіді у `.qmd` файлі. Графіки збережіть у `.py` файлі:

    ``` python
    plt.savefig('plot_error_vs_N.png', dpi=150, bbox_inches='tight')
    ```
    і покладіть PNG поруч із `.qmd` файлом.

3.  Відрендеруйте звіт:
    Для рендерингу звіту використовуйте Quarto. Встановіть його, якщо ще не зробили цього: https://quarto.org/docs/get-started/.

    ``` bash
    # HTML — не потребує LaTeX
    quarto render practices/practice_1/homework/report_Прізвище.qmd --to html

    # PDF — потребує LuaLaTeX (TeX Live або MiKTeX)
    quarto render practices/practice_1/homework/report_Прізвище.qmd --to pdf

> Якщо LaTeX не встановлений, спробуйте встановити, або збережіть pdf з браузера.