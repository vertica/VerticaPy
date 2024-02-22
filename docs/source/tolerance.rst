.. _tolerance:

==========================================
Tolerance with other popular ML algorithms
==========================================

At VerticaPy, we pride ourselves on delivering robust and 
reliable machine learning algorithms. As part of our 
commitment to quality assurance, we conduct rigorous unit 
tests to compare the performance of our algorithms with 
those from other popular open-source libraries, such as 
scikit-learn. These tests ensure that our implementations 
maintain parity with industry standards and deliver 
consistent results across various datasets. To provide 
transparency and facilitate easy reference, we establish 
specific tolerances for each type of algorithm. These 
tolerances serve as benchmarks for evaluating the 
accuracy and efficiency of our algorithms, guaranteeing 
that they meet or exceed expectations in diverse machine 
learning tasks.

.. math::

    relative tolerance = \frac{{|vpy\_score - py\_score|}}{{\min(|vpy\_score|, |py\_score|)}}


Below we have created a table which lists some of the different
tolerances for each model:

.. important:: All our models are tested against popular libraries, but here we only list a few. 

.. note:: 
    
    The formula for relative error is as follows:

    .. math::

        relative\ error = \frac{{|Score_{verticapy} - Score_{python}|}}{{\min(|Score_{verticapy}|, |Score_{python}|)}}

    The formula for absolute error is as follows:

    .. math::

        relative\ error = \frac{{|Score_{verticapy} - Score_{python}|}}{{1 + \min(|Score_{verticapy}|, |Score_{python}|)}}

    Absolute error is only used in cases where the result is very close to 0.

    For more information on pytest approxiamtions, please read `this link <https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-approx>`_

Regression
===========

.. ipython:: python
    :suppress:

    from verticapy.tests_new.machine_learning.vertica import rel_abs_tol_map

    def build_table(tols, table: str = ""):
        first_col_dashes = 30
        second_col_dashes = 22
        third_col_dashes = 22
        first_col_title = "Metric"
        second_col_title = "Relative Toelrance"
        third_col_title = "Absolute Tolerance"
        separators = "+" + "-" * first_col_dashes + "+" + "-" * second_col_dashes + "+" + "-" * third_col_dashes + "+\n"
        double_separators = separators.replace("-","=")
        table+= separators
        table+= f"| {first_col_title}" + " " * (first_col_dashes-len(first_col_title)-1) 
        table+= f"| {second_col_title}" + " " * (second_col_dashes-len(second_col_title)-1)
        table+= f"| {third_col_title}" + " " * (third_col_dashes-len(third_col_title)-1) + "|\n"
        table+= double_separators
        tols = dict(sorted(tols.items()))
        for key in tols.keys():
            table+= f"| {key}" + " " * (first_col_dashes-len(str(key))-1)
            table+= f"| {tols[key][0]}" + " " * (second_col_dashes-len(str(tols[key][0]))-1) 
            table+= f"| {tols[key][1]}" + " " * (third_col_dashes-len(str(tols[key][1]))-1) + "|\n"
            table+= separators
        return table

    def comparison_lib(key):
        if key in ["AR", "ARIMA", "MA", "ARMA"]:
            return "Statsmodel"
        elif key in ["TENSORFLOW"]:
            return "Tensorflow"
        return "Scikit Learn"

    keys_list = list(rel_abs_tol_map.keys())
    keys_list.sort()
    table = ""
    for algorithm in keys_list:
        included_terms = ['Regr', 'Ridge', 'Lasso', 'Elastic', 'SVR']
        if any(term in algorithm for term in included_terms):
            details = rel_abs_tol_map[algorithm] 
            table += algorithm + "\n" + "-" * len(algorithm) + "\n"
            table += "**Comparison Library**:" + " :bdg-primary-line:`" + comparison_lib(algorithm) + "`" + "\n" + "\n"
            tols = {key: (details[key]['rel'], details[key]['abs']) for key in details.keys()}
            tols
            table = build_table(tols, table)
            table += "\n"
    file_path = "tolerance_table_regression.rst"
    with open(file_path, "w") as rst_file:
        rst_file.write(table)

.. include:: ../tolerance_table_regression.rst

Classification
===============

.. ipython:: python
    :suppress:

    keys_list = list(rel_abs_tol_map.keys())
    keys_list.sort()
    table = ""
    for algorithm in keys_list:
        included_terms = ['Classifier']
        if any(term in algorithm for term in included_terms):
            details = rel_abs_tol_map[algorithm] 
            table += algorithm + "\n" + "-" * len(algorithm) + "\n"
            table += "**Comparison Library**:" + " :bdg-primary-line:`" + comparison_lib(algorithm) + "`" + "\n" + "\n"
            tols = {key: (details[key]['rel'], details[key]['abs']) for key in details.keys()}
            tols
            table = build_table(tols, table)
            table += "\n"
    file_path = "tolerance_table_classification.rst"
    with open(file_path, "w") as rst_file:
        rst_file.write(table)

.. include:: ../tolerance_table_classification.rst

Others
=======

.. ipython:: python
    :suppress:

    keys_list = list(rel_abs_tol_map.keys())
    keys_list.sort()
    table = ""
    for algorithm in keys_list:
        excluded_terms = ['Regr', 'Classifier', 'Ridge', 'Lasso', 'Elastic', 'SVR', 'TENSORFLOW', 'TF']
        if not any(term in algorithm for term in excluded_terms):
            details = rel_abs_tol_map[algorithm] 
            table += algorithm + "\n" + "-" * len(algorithm) + "\n"
            table += "**Comparison Library**:" + " :bdg-primary-line:`" + comparison_lib(algorithm) + "`" + "\n" + "\n"
            tols = {key: (details[key]['rel'], details[key]['abs']) for key in details.keys()}
            tols
            table = build_table(tols, table)
            table += "\n"
    file_path = "tolerance_table_others.rst"
    with open(file_path, "w") as rst_file:
        rst_file.write(table)
    


.. include:: ../tolerance_table_others.rst

.. note:: To learn more about the subtle difference betweeen Absolute Error and Relative Error please refer to `this link <https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-approx>`_.

