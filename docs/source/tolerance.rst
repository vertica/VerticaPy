.. _tolerance:

======================================
Tolerance's with other ML algorithms
======================================

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

Below we have created a table which lists some of the different
tolerances for each model:

.. important:: All our models are tested against popular libraries, but here we only list a few. 

.. ipython:: python
    :suppress:

    from verticapy.tests_new.machine_learning.vertica import rel_tolerance_map, abs_tolerance_map

    first_col_dashes = 30
    second_col_dashes = 20
    third_col_dashes = 20
    fourth_col_dashes = 20

    first_col_title = "Algorithm"
    second_col_title = "Relative Toelrance"
    third_col_title = "Absolute Tolerance"
    fourth_col_title = "Comparison Library"
    scikit = "Scikit Learn"
    stats = "Statsmodels"
    tensorflow = "Tensorflow"
    table = ""
    separators = "+" + "-" * first_col_dashes + "+" + "-" * second_col_dashes + "+" + "-" * third_col_dashes + "+" + "-" * fourth_col_dashes + "+\n"
    double_separators = separators.replace("-","=")
    table+= separators
    table+= f"| {first_col_title}" + " " * (first_col_dashes-len(first_col_title)-1) 
    table+= f"| {second_col_title}" + " " * (second_col_dashes-len(second_col_title)-1)
    table+= f"| {third_col_title}" + " " * (third_col_dashes-len(third_col_title)-1)
    table+= f"| {fourth_col_title}" + " " * (fourth_col_dashes-len(fourth_col_title)-1) + "|\n"
    table+= double_separators
    tols = {key: (rel_tolerance_map[key], abs_tolerance_map[key]) for key in rel_tolerance_map.keys()}
    tols = dict(sorted(tols.items()))
    for key in tols.keys():
        table+= f"| {key}" + " " * (first_col_dashes-len(key)-1) 
        table+= f"| {tols[key][0]}" + " " * (second_col_dashes-len(str(tols[key][0]))-1)
        table+= f"| {tols[key][1]}" + " " * (third_col_dashes-len(str(tols[key][1]))-1)
        if key in ["AR", "ARIMA", "MA", "ARMA"]:
            table+= f"| {stats}" + " " * (third_col_dashes-len(stats)-1) + "|\n"
        elif key in ["TENSORFLOW"]:
            table+= f"| {tensorflow}" + " " * (third_col_dashes-len(tensorflow)-1) + "|\n"
        else:
            table+= f"| {scikit}" + " " * (third_col_dashes-len(scikit)-1) + "|\n"
        table+= separators
    file_path = "tolerance_table.rst"
    with open(file_path, "w") as rst_file:
        rst_file.write(table)

.. include:: ../tolerance_table.rst

.. note:: To learn more about the subtle difference betweeen Absolute Error and Relative Error please refer to `this link <https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-approx>`_.

