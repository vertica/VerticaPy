<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="RandomForestRegressor">RandomForestRegressor<a class="anchor-link" href="#RandomForestRegressor">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">RandomForestRegressor</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                      <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
                      <span class="n">n_estimators</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">10</span><span class="p">,</span>
                      <span class="n">max_features</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                      <span class="n">max_leaf_nodes</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mf">1e9</span><span class="p">,</span> 
                      <span class="n">sample</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.632</span><span class="p">,</span>
                      <span class="n">max_depth</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                      <span class="n">min_samples_leaf</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span>
                      <span class="n">min_info_gain</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                      <span class="n">nbins</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">32</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a RandomForestRegressor object by using the Vertica Highly Distributed and Scalable Random Forest on the data. It is one of the ensemble learning method for regression that operate by constructing a multitude of decision trees at training time and outputting the mean prediction.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the model to be stored in the database.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">n_estimators</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The number of trees in the forest, an integer between 0 and 1000, inclusive.</td> </tr>
    <tr> <td><div class="param_name">max_features</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>The number of randomly chosen features from which to pick the best feature to split on a given tree node. It can be an integer or one of the two following methods.<br><ul>
                                                        <li><b>auto :</b> square root of the total number of predictors.</li>
                                                        <li><b>max :</b> number of predictors.</li></ul></td> </tr>
    <tr> <td><div class="param_name">max_leaf_nodes</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The maximum number of leaf nodes a tree in the forest can have, an integer between 1 and 1e9, inclusive.</td> </tr>
    <tr> <td><div class="param_name">sample</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>The portion of the input data set that is randomly picked for training each tree, a float between 0.0 and 1.0, inclusive.</td> </tr>
    <tr> <td><div class="param_name">max_depth</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The maximum depth for growing each tree, an integer between 1 and 100, inclusive.</td> </tr>
    <tr> <td><div class="param_name">min_samples_leaf</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The minimum number of samples each branch must have after splitting a node, an integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples is discarded.</td> </tr>
    <tr> <td><div class="param_name">min_info_gain</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. A split with information gain less than this threshold is discarded.</td> </tr>
    <tr> <td><div class="param_name">nbins</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The number of bins to use for continuous features, an integer between 2 and 1000, inclusive.</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
    <tr> <td><div class="param_name">y</div></td> <td><div class="type">str</div></td> <td>Response column.</td> </tr>
    <tr> <td><div class="param_name">test_relation</div></td> <td><div class="type">float</div></td> <td>Relation used to test the model. All the model methods are abstractions which will simplify the process. The test relation will be used by many methods to evaluate the model. If empty, the training relation will be used as test. You can change it anytime by changing the test_relation attribute of the object.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Regression/deploySQL/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Regression/drop/index.php">drop</a></td> <td>Drops the model from the Vertica DB.</td> </tr>
    <tr> <td><a href="../Regression/export_graphviz/index.php">export_graphviz</a></td> <td>Converts the input tree to graphviz.</td> </tr>
    <tr> <td><a href="../Regression/features_importance/index.php">features_importance</a></td> <td>Computes the model features importance using the Gini Index.</td> </tr>
    <tr> <td><a href="../Regression/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Regression/get_tree/index.php">get_tree</a></td> <td>Returns a tablesample with all the input tree information.</td> </tr>
    <tr> <td><a href="../Regression/plot_tree/index.php">plot_tree</a></td> <td>Draws the input tree. The module anytree must be installed in the machine.</td> </tr>
    <tr> <td><a href="../Regression/predict/index.php">predict</a></td> <td>Predicts using the input relation.</td> </tr>
    <tr> <td><a href="../Regression/regression_report/index.php">regression_report</a></td> <td>Computes a regression report using multiple metrics to evaluate the model (r2, mse, max error...). </td> </tr>
    <tr> <td><a href="../Regression/score/index.php">score</a></td> <td>Computes the model score.</td> </tr>

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
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestRegressor</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestRegressor</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.rf_winequality&quot;</span><span class="p">,</span>
                              <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span>
                              <span class="n">max_features</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                              <span class="n">max_leaf_nodes</span> <span class="o">=</span> <span class="mi">32</span><span class="p">,</span> 
                              <span class="n">sample</span> <span class="o">=</span> <span class="mf">0.7</span><span class="p">,</span>
                              <span class="n">max_depth</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span>
                              <span class="n">min_samples_leaf</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                              <span class="n">min_info_gain</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                              <span class="n">nbins</span> <span class="o">=</span> <span class="mi">32</span><span class="p">)</span>
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
<pre>&lt;RandomForestRegressor&gt;
</pre>
</div>
</div>

</div>
</div>

</div>