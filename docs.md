
## Як рендерити презентації


``` bash
quarto preview lectures/lecture_4/presentation.qmd
quarto preview lectures/lecture_5/presentation.qmd
quarto render lectures/lecture_4/presentation.qmd --to pdf
quarto render lectures/lecture_4/presentation.qmd --to beamer
quarto preview lectures/lecture_7/presentation7.qmd
quarto preview lectures/lecture_7/lecture_notes_7.qmd
```


## render wiki
cd /home/hronos/code/intro2agi_course
quarto preview wiki/wiki.qmd
quarto render practices/practice_3/homework/instructions3.qmd --to beamer