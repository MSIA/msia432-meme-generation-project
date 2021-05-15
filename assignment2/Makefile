.PHONY: to_notebook to_script

assignment2b_v2.ipynb:
	jupytext --to notebook assignment2b_v2.py

assignment2b_v2.py:
	jupytext --to py:percent --opt comment_magics=false assignment2b_v2.ipynb

to_notebook: assignment2b_v2.ipynb
to_script: assignment2b_v2.py
