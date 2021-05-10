Currently, I rely on using `jupytext` to do version control for notebooks. This process is kind of tedious but it might be helpful in this debugging stage. We can definitely switch to other methods to do version control after now. You can install by using `pip install jupytext`. To learn more details about this package, you can their [github page](https://github.com/mwouts/jupytext).

Now after you getting the .py file from this repo, you can convert it back to a notebook with the following command.

```bash
jupytext --to notebook assignment2b_v2.py
```

After editing the notebook, you can use the following command to convert the notebook back to the .py file. 

```bash
jupytext --to py:percent --opt comment_magics=false notebook.ipynb
```

You can see a full list of command in [here](https://jupytext.readthedocs.io/en/latest/using-cli.html).

You can also use the sync mode to update the .py and .ipynb file with the following command.

```bash
 jupytext --set-formats ipynb,py:percent assignment2b_v2.ipynb
```

But be careful, don't open and edit the .py and .ipynb at the same time. Learn more about sync mode in [here](https://github.com/mwouts/jupytext#paired-notebooks).

