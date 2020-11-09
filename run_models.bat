@ECHO OFF
:LSTM+Wiki
::ECHO LSTM + Wikitext2 + Bias Reg 0.0
::python ./main.py --model lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 0.0 --lr 20 --cuda
::ECHO LSTM + Wikitext2 + Bias Reg 0.01
::python ./main.py --model lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 0.01 --lr 20 --cuda
::ECHO LSTM + Wikitext2 + Bias Reg 0.1
::python ./main.py --model lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 0.1 --lr 20 --cuda
::ECHO LSTM + Wikitext2 + Bias Reg 0.5
::python ./main.py --model lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 0.5 --lr 20 --cuda
::ECHO LSTM + Wikitext2 + Bias Reg 1.0
::python ./main.py --model lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 1 --lr 20 --cuda

:LSTM+BR_lit
ECHO LSTM + BR_lit + Bias Reg 0.0
python ./main.py --model lstm --dataset br_lit --bias_reg --bias_reg_lambda 0.0 --lr 20 --cuda
ECHO LSTM + BR_lit + Bias Reg 0.01
python ./main.py --model lstm --dataset br_lit --bias_reg --bias_reg_lambda 0.01 --lr 20 --cuda
ECHO LSTM + BR_lit + Bias Reg 0.1
python ./main.py --model lstm --dataset br_lit --bias_reg --bias_reg_lambda 0.1 --lr 20 --cuda
ECHO LSTM + BR_lit + Bias Reg 0.20
python ./main.py --model lstm --dataset br_lit --bias_reg --bias_reg_lambda=0.5 --lr 20 --cuda
ECHO LSTM + BR_lit + Bias Reg 1.0
python ./main.py --model lstm --dataset br_lit --bias_reg --bias_reg_lambda=1 --lr 20 --cuda

:LSTM+Blogset
ECHO LSTM + Blogset + Bias Reg 0.0
python ./main.py --model lstm --dataset blogset --bias_reg --bias_reg_lambda 0.0 --lr 20 --cuda
ECHO LSTM + Blogset + Bias Reg 0.01
python ./main.py --model lstm --dataset blogset --bias_reg --bias_reg_lambda 0.01 --lr 20 --cuda
ECHO LSTM + Blogset + Bias Reg 0.1
python ./main.py --model lstm --dataset blogset --bias_reg --bias_reg_lambda 0.1 --lr 20 --cuda
ECHO LSTM + Blogset + Bias Reg 0.20
python ./main.py --model lstm --dataset blogset --bias_reg --bias_reg_lambda 0.5 --lr 20 --cuda
ECHO LSTM + Blogset + Bias Reg 1.0
python ./main.py --model lstm --dataset blogset --bias_reg --bias_reg_lambda 1 --lr 20 --cuda

:ATT_LSTM+Wiki
ECHO ATT_LSTM + Wikitext2 + Bias Reg 0.0
python ./main.py --model att_lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 0.0 --lr 20 --cuda
ECHO ATT_LSTM + Wikitext2 + Bias Reg 0.01
python ./main.py --model att_lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 0.01 --lr 20 --cuda
ECHO ATT_LSTM + Wikitext2 + Bias Reg 0.1
python ./main.py --model att_lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 0.1 --lr 20 --cuda
ECHO ATT_LSTM + Wikitext2 + Bias Reg 0.5
python ./main.py --model att_lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 0.5 --lr 20 --cuda
ECHO ATT_LSTM + Wikitext2 + Bias Reg 1.0
python ./main.py --model att_lstm --dataset wikitext2 --bias_reg --bias_reg_lambda 1 --lr 20 --cuda

:ATT_LSTM+BR_lit
ECHO ATT_LSTM + BR_lit + Bias Reg 0.0
python ./main.py --model att_lstm --dataset br_lit --bias_reg --bias_reg_lambda 0.0 --lr 20 --cuda
ECHO ATT_LSTM + BR_lit + Bias Reg 0.01
python ./main.py --model att_lstm --dataset br_lit --bias_reg --bias_reg_lambda 0.01 --lr 20 --cuda
ECHO ATT_LSTM + BR_lit + Bias Reg 0.1
python ./main.py --model att_lstm --dataset br_lit --bias_reg --bias_reg_lambda 0.1 --lr 20 --cuda
ECHO ATT_LSTM + BR_lit + Bias Reg 0.5
python ./main.py --model att_lstm --dataset br_lit --bias_reg --bias_reg_lambda=0.5 --lr 20 --cuda
ECHO ATT_LSTM + BR_lit + Bias Reg 1.0
python ./main.py --model att_lstm --dataset br_lit --bias_reg --bias_reg_lambda=1 --lr 20 --cuda

:ATT_LSTM+Blogset
ECHO ATT_LSTM + Blogset + Bias Reg 0.0
python ./main.py --model att_lstm --dataset blogset --bias_reg --bias_reg_lambda 0.0 --lr 20 --cuda
ECHO ATT_LSTM + Blogset + Bias Reg 0.01
python ./main.py --model att_lstm --dataset blogset --bias_reg --bias_reg_lambda 0.01 --lr 20 --cuda
ECHO ATT_LSTM + Blogset + Bias Reg 0.1
python ./main.py --model att_lstm --dataset blogset --bias_reg --bias_reg_lambda 0.1 --lr 20 --cuda
ECHO ATT_LSTM + Blogset + Bias Reg 0.5
python ./main.py --model att_lstm --dataset blogset --bias_reg --bias_reg_lambda 0.5 --lr 20 --cuda
ECHO ATT_LSTM + Blogset + Bias Reg 1.0
python ./main.py --model att_lstm --dataset blogset --bias_reg --bias_reg_lambda 1 --lr 20 --cuda

:TRANSFORMER+Wiki
ECHO TRANSFORMER + Wikitext2 + Bias Reg 0.0
python ./main.py --model transformer --dataset wikitext2 --bias_reg --bias_reg_lambda 0.0 --lr 5 --cuda
ECHO TRANSFORMER + Wikitext2 + Bias Reg 0.01
python ./main.py --model transformer --dataset wikitext2 --bias_reg --bias_reg_lambda 0.01 --lr 5 --cuda
ECHO TRANSFORMER + Wikitext2 + Bias Reg 0.1
python ./main.py --model transformer --dataset wikitext2 --bias_reg --bias_reg_lambda 0.1 --lr 5 --cuda
ECHO TRANSFORMER + Wikitext2 + Bias Reg 0.5
python ./main.py --model transformer --dataset wikitext2 --bias_reg --bias_reg_lambda 0.5 --lr 5 --cuda
ECHO TRANSFORMER + Wikitext2 + Bias Reg 1.0
python ./main.py --model transformer --dataset wikitext2 --bias_reg --bias_reg_lambda 1 --lr 5 --cuda

:TRANSFORMER+BR_lit
ECHO TRANSFORMER + BR_lit + Bias Reg 0.0
python ./main.py --model transformer --dataset br_lit --bias_reg --bias_reg_lambda 0.0 --lr 5 --cuda
ECHO TRANSFORMER + BR_lit + Bias Reg 0.01
python ./main.py --model transformer --dataset br_lit --bias_reg --bias_reg_lambda 0.01 --lr 5 --cuda
ECHO TRANSFORMER + BR_lit + Bias Reg 0.1
python ./main.py --model transformer --dataset br_lit --bias_reg --bias_reg_lambda 0.1 --lr 5 --cuda
ECHO TRANSFORMER + BR_lit + Bias Reg 0.5
python ./main.py --model transformer --dataset br_lit --bias_reg --bias_reg_lambda=0.5 --lr 5 --cuda
ECHO TRANSFORMER + BR_lit + Bias Reg 1.0
python ./main.py --model transformer --dataset br_lit --bias_reg --bias_reg_lambda=1 --lr 5 --cuda

:TRANSFORMER+Blogset
ECHO TRANSFORMER + Blogset + Bias Reg 0.0
python ./main.py --model transformer --dataset blogset --bias_reg --bias_reg_lambda 0.0 --lr 5 --cuda
ECHO TRANSFORMER + Blogset + Bias Reg 0.01
python ./main.py --model transformer --dataset blogset --bias_reg --bias_reg_lambda 0.01 --lr 5 --cuda
ECHO TRANSFORMER + Blogset + Bias Reg 0.1
python ./main.py --model transformer --dataset blogset --bias_reg --bias_reg_lambda 0.1 --lr 5 --cuda
ECHO TRANSFORMER + Blogset + Bias Reg 0.5
python ./main.py --model transformer --dataset blogset --bias_reg --bias_reg_lambda 0.5 --lr 5 --cuda
ECHO TRANSFORMER + Blogset + Bias Reg 1.0
python ./main.py --model transformer --dataset blogset --bias_reg --bias_reg_lambda 1 --lr 5 --cuda
:end
PAUSE