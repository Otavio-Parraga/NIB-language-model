::GOTO transformer
:lstm
python ./evaluate.py --model lstm --dataset wikitext2 --bias_reg 0.0 --split test
python ./evaluate.py --model lstm --dataset wikitext2 --bias_reg 0.01 --split test
python ./evaluate.py --model lstm --dataset wikitext2 --bias_reg 0.1 --split test
python ./evaluate.py --model lstm --dataset wikitext2 --bias_reg 0.5 --split test
python ./evaluate.py --model lstm --dataset wikitext2 --bias_reg 1.0 --split test

python ./evaluate.py --model lstm --dataset br_lit --bias_reg 0.0 --split test
python ./evaluate.py --model lstm --dataset br_lit --bias_reg 0.01 --split test
python ./evaluate.py --model lstm --dataset br_lit --bias_reg 0.1 --split test
python ./evaluate.py --model lstm --dataset br_lit --bias_reg=0.5 --split test
python ./evaluate.py --model lstm --dataset br_lit --bias_reg 1.0 --split test

python ./evaluate.py --model lstm --dataset blogset --bias_reg 0.0 --split test
python ./evaluate.py --model lstm --dataset blogset --bias_reg 0.01 --split test
python ./evaluate.py --model lstm --dataset blogset --bias_reg 0.1 --split test
python ./evaluate.py --model lstm --dataset blogset --bias_reg 0.5 --split test
python ./evaluate.py --model lstm --dataset blogset --bias_reg 1.0 --split test
:att_lstm
python ./evaluate.py --model att_lstm --dataset wikitext2 --bias_reg 0.0 --split test
python ./evaluate.py --model att_lstm --dataset wikitext2 --bias_reg 0.01 --split test
python ./evaluate.py --model att_lstm --dataset wikitext2 --bias_reg 0.1 --split test
python ./evaluate.py --model att_lstm --dataset wikitext2 --bias_reg 0.5 --split test
python ./evaluate.py --model att_lstm --dataset wikitext2 --bias_reg 1.0 --split test

python ./evaluate.py --model att_lstm --dataset br_lit --bias_reg 0.0 --split test
python ./evaluate.py --model att_lstm --dataset br_lit --bias_reg 0.01 --split test
python ./evaluate.py --model att_lstm --dataset br_lit --bias_reg 0.1 --split test
python ./evaluate.py --model att_lstm --dataset br_lit --bias_reg=0.5 --split test
python ./evaluate.py --model att_lstm --dataset br_lit --bias_reg 1.0 --split test

python ./evaluate.py --model att_lstm --dataset blogset --bias_reg 0.0 --split test
python ./evaluate.py --model att_lstm --dataset blogset --bias_reg 0.01 --split test
python ./evaluate.py --model att_lstm --dataset blogset --bias_reg 0.1 --split test
python ./evaluate.py --model att_lstm --dataset blogset --bias_reg 0.5 --split test
python ./evaluate.py --model att_lstm --dataset blogset --bias_reg 1.0 --split test
:transformer
python ./evaluate.py --model transformer --dataset wikitext2 --bias_reg 0.0 --split test
python ./evaluate.py --model transformer --dataset wikitext2 --bias_reg 0.01 --split test
python ./evaluate.py --model transformer --dataset wikitext2 --bias_reg 0.1 --split test
python ./evaluate.py --model transformer --dataset wikitext2 --bias_reg 0.5 --split test
python ./evaluate.py --model transformer --dataset wikitext2 --bias_reg 1.0 --split test

python ./evaluate.py --model transformer --dataset br_lit --bias_reg 0.0 --split test
python ./evaluate.py --model transformer --dataset br_lit --bias_reg 0.01 --split test
python ./evaluate.py --model transformer --dataset br_lit --bias_reg 0.1 --split test
python ./evaluate.py --model transformer --dataset br_lit --bias_reg=0.5 --split test
python ./evaluate.py --model transformer --dataset br_lit --bias_reg 1.0 --split test

python ./evaluate.py --model transformer --dataset blogset --bias_reg 0.0 --split test
python ./evaluate.py --model transformer --dataset blogset --bias_reg 0.01 --split test
python ./evaluate.py --model transformer --dataset blogset --bias_reg 0.1 --split test
python ./evaluate.py --model transformer --dataset blogset --bias_reg 0.5 --split test
python ./evaluate.py --model transformer --dataset blogset --bias_reg 1.0 --split test
PAUSE