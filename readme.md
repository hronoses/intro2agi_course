# Матеріали до курсу "Вступ до штучного загального інтелекту"

Цей репозиторій містить матеріали до курсу "Вступ до загального штучного інтелекту": лекційні презентації, базовий код та шаблони звітів для практичних завдань.

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

Готовий HTML відкривається без Quarto — просто у браузері:

``` bash
xdg-open lectures/lecture_4/presentation.html   # Linux
open lectures/lecture_4/presentation.html        # macOS
```

Або live-preview з автооновленням (потребує [Quarto](https://quarto.org/docs/get-started/)):

``` bash
quarto preview lectures/lecture_4/presentation.qmd
quarto render lectures/lecture_4/presentation.qmd --to pdf
quarto render lectures/lecture_4/presentation.qmd --to beamer
```

## Домашні завдання

### Запустіть стартовий код та ознайомтесь із середовищем

``` bash
python practice_1/1_binary_discrete.py
```

### Переглянути умову завдання

Pdf версія умови завдання знаходиться за адресою `practice_1/homework/instructions.pdf`.

### Заповнити та здати звіт

1.  Скопіюйте шаблон під своє прізвище:

    ``` bash
    cp practice_1/homework/report_template.qmd practice_1/homework/report_Прізвище.qmd
    ```

2.  Заповніть відповіді у `.qmd` файлі. Графіки збережіть у `.py` файлі:

    ``` python
    plt.savefig('plot_error_vs_N.png', dpi=150, bbox_inches='tight')
    ```

    і покладіть PNG поруч із `.qmd` файлом.

3.  Відрендеруйте звіт:

    ``` bash
    # HTML — не потребує LaTeX
    quarto render practice_1/homework/report_Прізвище.qmd --to html

    # PDF — потребує LuaLaTeX (TeX Live або MiKTeX)
    quarto render practice_1/homework/report_Прізвище.qmd --to pdf
    quarto render practice_1/homework/instructions.qmd --to pdf
    ```

> Якщо LaTeX не встановлений, спробуйте встановити, або збережіть pdf з браузера.