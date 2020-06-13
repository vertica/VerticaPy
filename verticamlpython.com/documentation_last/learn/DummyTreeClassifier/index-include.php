<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="DummyTreeClassifier">DummyTreeClassifier<a class="anchor-link" href="#DummyTreeClassifier">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">DummyTreeClassifier</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                    <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Dummy Tree Classifier. This classifier learns by heart the training data =&gt; very depth RandomForestClassifier of one tree using all the data.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the the model. The model will be stored in the DB.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">classes</div></td> <td><div class="type">list</div></td>  <td>List of all the response classes.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td>  <td>List of the predictors.</td> </tr>
    <tr> <td><div class="param_name">y</div></td> <td><div class="type">str</div></td> <td>Response column.</td> </tr>
    <tr> <td><div class="param_name">test_relation</div></td> <td><div class="type">float</div></td> <td>Relation used to test the model. All the model methods are abstractions which will simplify the process. The test relation will be used by many methods to evaluate the model. If empty, the train relation will be used as test. You can change it anytime by changing the test_relation attribute of the object.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Classification/classification_report/index.php">classification_report</a></td> <td>Computes a classification report using multiple metrics to evaluate the model (AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will consider each category as positive and switch to the next one during the computation.</td> </tr>
    <tr> <td><a href="../Classification/confusion_matrix/index.php">confusion_matrix</a></td> <td>Computes the model confusion matrix.</td> </tr>
    <tr> <td><a href="../Classification/deploySQL/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Classification/drop/index.php">drop</a></td> <td>Drops the model from the Vertica DB.</td> </tr>
    <tr> <td><a href="../Classification/export_graphviz/index.php">export_graphviz</a></td> <td>Converts the input tree to graphviz.</td> </tr>
    <tr> <td><a href="../Classification/features_importance/index.php">features_importance</a></td> <td>Computes the model features importance using the Gini Index.</td> </tr>
    <tr> <td><a href="../Classification/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Classification/get_tree/index.php">get_tree</a></td> <td>Returns a tablesample with all the input tree information.</td> </tr>
    <tr> <td><a href="../Classification/lift_chart/index.php">lift_chart</a></td> <td>Draws the model Lift Chart.</td> </tr>
    <tr> <td><a href="../Classification/plot_tree/index.php">plot_tree</a></td> <td>Draws the input tree. The module anytree must be installed in the machine.</td> </tr>
    <tr> <td><a href="../Classification/prc_curve/index.php">prc_curve</a></td> <td>Draws the model PRC curve.</td> </tr>
    <tr> <td><a href="../Classification/predict/index.php">predict</a></td> <td>Predicts using the input relation.</td> </tr>
    <tr> <td><a href="../Classification/roc_curve/index.php">roc_curve</a></td> <td>Draws the model ROC curve.</td> </tr>
    <tr> <td><a href="../Classification/score/index.php">score</a></td> <td>Computes the model score.</td> </tr>

</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.tree</span> <span class="k">import</span> <span class="n">DummyTreeClassifier</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">DummyTreeClassifier</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.rf_titanic&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;RandomForestClassifier&gt;
</pre>
</div>
</div>

</div>
</div>

</div>