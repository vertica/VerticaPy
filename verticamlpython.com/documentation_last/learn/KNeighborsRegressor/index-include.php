<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="KNeighborsRegressor">KNeighborsRegressor (Beta)<a class="anchor-link" href="#KNeighborsRegressor">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">KNeighborsRegressor</span><span class="p">(</span><span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
                    <span class="n">n_neighbors</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                    <span class="n">p</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a KNeighborsRegressor object by using the K Nearest Neighbors Algorithm. This object is using pure SQL to compute all the distances and final score. It is using CROSS JOIN and may be really expensive in some cases. As KNeighborsRegressor is using the p-distance, it is highly sensible to un-normalized data.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">n_neighbors</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Number of neighbors to consider when computing the score.</td> </tr>
    <tr> <td><div class="param_name">p</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The p corresponding to the one of the p-distance (distance metric used during the model computation).</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
    <tr> <td><div class="param_name">y</div></td> <td><div class="type">str</div></td> <td>Response column.</td> </tr>
    <tr> <td><div class="param_name">test_relation</div></td> <td><div class="type">float</div></td> <td>Relation used to test the model. All the model methods are abstractions which will simplify the process. The test relation will be used by many methods to evaluate the model. If empty, the train relation will be used as test. You can change it anytime by changing the test_relation attribute of the object.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Regression/deploySQL/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Regression/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Regression/regression_report/index.php">regression_report</a></td> <td>Computes a regression report using multiple metrics to evaluate the model (r2, mse, max error...). </td> </tr>
    <tr> <td><a href="../Regression/score/index.php">score</a></td> <td>Computes the model score.</td> </tr>
    <tr> <td><a href="../Regression/to_vdf/index.php">to_vdf</a></td> <td>Returns the vDataFrame of the Prediction.</td> </tr>

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
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.neighbors</span> <span class="k">import</span> <span class="n">KNeighborsRegressor</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">KNeighborsRegressor</span><span class="p">(</span><span class="n">n_neighbors</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                            <span class="n">p</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
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
<pre>&lt;KNeighborsRegressor&gt;
</pre>
</div>
</div>

</div>
</div>

</div>