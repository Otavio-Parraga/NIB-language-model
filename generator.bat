@echo OFF
:lstm
::python ./generate.py --model lstm --dataset wikitext2 --bias_reg 0.0 --number 1300
::python ./generate.py --model lstm --dataset wikitext2 --bias_reg 0.01 --number 1300
::python ./generate.py --model lstm --dataset wikitext2 --bias_reg 0.1 --number 1300
::python ./generate.py --model lstm --dataset wikitext2 --bias_reg 0.5 --number 1300
::python ./generate.py --model lstm --dataset wikitext2 --bias_reg 1.0 --number 1300

python ./generate.py --model lstm --dataset br_lit --bias_reg 0.0 --number 1300
python ./generate.py --model lstm --dataset br_lit --bias_reg 0.01 --number 1300
python ./generate.py --model lstm --dataset br_lit --bias_reg 0.1 --number 1300
python ./generate.py --model lstm --dataset br_lit --bias_reg 0.5 --number 1300
python ./generate.py --model lstm --dataset br_lit --bias_reg 1.0 --number 1300

python ./generate.py --model lstm --dataset blogset --bias_reg 0.0 --number 1300
python ./generate.py --model lstm --dataset blogset --bias_reg 0.01 --number 1300
python ./generate.py --model lstm --dataset blogset --bias_reg 0.1 --number 1300
python ./generate.py --model lstm --dataset blogset --bias_reg 0.5 --number 1300
python ./generate.py --model lstm --dataset blogset --bias_reg 1.0 --number 1300

:att_lstm
python ./generate.py --model att_lstm --dataset wikitext2 --bias_reg 0.0 --number 1300
python ./generate.py --model att_lstm --dataset wikitext2 --bias_reg 0.01 --number 1300
python ./generate.py --model att_lstm --dataset wikitext2 --bias_reg 0.1 --number 1300
python ./generate.py --model att_lstm --dataset wikitext2 --bias_reg 0.5 --number 1300
python ./generate.py --model att_lstm --dataset wikitext2 --bias_reg 1.0 --number 1300

python ./generate.py --model att_lstm --dataset br_lit --bias_reg 0.0 --number 1300
python ./generate.py --model att_lstm --dataset br_lit --bias_reg 0.01 --number 1300
python ./generate.py --model att_lstm --dataset br_lit --bias_reg 0.1 --number 1300
python ./generate.py --model att_lstm --dataset br_lit --bias_reg 0.5 --number 1300
python ./generate.py --model att_lstm --dataset br_lit --bias_reg 1.0 --number 1300

python ./generate.py --model att_lstm --dataset blogset --bias_reg 0.0 --number 1300
python ./generate.py --model att_lstm --dataset blogset --bias_reg 0.01 --number 1300
python ./generate.py --model att_lstm --dataset blogset --bias_reg 0.1 --number 1300
python ./generate.py --model att_lstm --dataset blogset --bias_reg 0.5 --number 1300
python ./generate.py --model att_lstm --dataset blogset --bias_reg 1.0 --number 1300

:transformer
python ./generate.py --model transformer --dataset wikitext2 --bias_reg 0.0 --number 1300
python ./generate.py --model transformer --dataset wikitext2 --bias_reg 0.01 --number 1300
python ./generate.py --model transformer --dataset wikitext2 --bias_reg 0.1 --number 1300
python ./generate.py --model transformer --dataset wikitext2 --bias_reg 0.5 --number 1300
python ./generate.py --model transformer --dataset wikitext2 --bias_reg 1.0 --number 1300

python ./generate.py --model transformer --dataset br_lit --bias_reg 0.0 --number 1300
python ./generate.py --model transformer --dataset br_lit --bias_reg 0.01 --number 1300
python ./generate.py --model transformer --dataset br_lit --bias_reg 0.1 --number 1300
python ./generate.py --model transformer --dataset br_lit --bias_reg 0.5 --number 1300
python ./generate.py --model transformer --dataset br_lit --bias_reg 1.0 --number 1300

python ./generate.py --model transformer --dataset blogset --bias_reg 0.0 --number 1300
python ./generate.py --model transformer --dataset blogset --bias_reg 0.01 --number 1300
python ./generate.py --model transformer --dataset blogset --bias_reg 0.1 --number 1300
python ./generate.py --model transformer --dataset blogset --bias_reg 0.5 --number 1300
python ./generate.py --model transformer --dataset blogset --bias_reg 1.0 --number 1300
:end
PAUSE