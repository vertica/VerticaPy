<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="NearestCentroid">NearestCentroid (Beta)<a class="anchor-link" href="#NearestCentroid">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">NearestCentroid</span><span class="p">(</span><span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
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
<p>Creates a NearestCentroid object by using the K Nearest Centroid Algorithm. This object is using pure SQL to compute all the distances and final score. As NearestCentroid is using the p-distance, it is highly sensible to un-normalized data.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">p</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The p corresponding to the one of the p-distance (distance metric used during the model computation).</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">centroids</div></td> <td><div class="type">tablesample</div></td> <td>The final centroids.</td> </tr>
    <tr> <td><div class="param_name">classes</div></td> <td><div class="type">list</div></td> <td>List of all the response classes.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
    <tr> <td><div class="param_name">y</div></td> <td><div class="type">str</div></td> <td>Response column.</td> </tr>
    <tr> <td><div class="param_name">test_relation</div></td> <td><div class="type">float</div></td> <td>Relation used to test the model. All the model methods are abstractions which will simplify the process. The test relation will be used by many methods to evaluate the model. If empty, the train relation will be used as test. You can change it anytime by changing the test_relation attribute of the object.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Classification/classification_report/index.php">classification_report</a></td> <td>Computes a classification report using multiple metrics to evaluate the model (AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will consider each category as positive and switch to the next one during the computation.</td> </tr>
    <tr> <td><a href="../Classification/confusion_matrix/index.php">confusion_matrix</a></td> <td>Computes the model confusion matrix.</td> </tr>
    <tr> <td><a href="../Classification/deploySQL/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Classification/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Classification/lift_chart/index.php">lift_chart</a></td> <td>Draws the model Lift Chart.</td> </tr>
    <tr> <td><a href="../Classification/prc_curve/index.php">prc_curve</a></td> <td>Draws the model PRC curve.</td> </tr>
    <tr> <td><a href="../Classification/roc_curve/index.php">roc_curve</a></td> <td>Draws the model ROC curve.</td> </tr>
    <tr> <td><a href="../Classification/score/index.php">score</a></td> <td>Computes the model score.</td> </tr>
    <tr> <td><a href="../Classification/to_vdf/index.php">to_vdf</a></td> <td>Returns the vDataFrame of the Prediction.</td> </tr>

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
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.neighbors</span> <span class="k">import</span> <span class="n">NearestCentroid</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">NearestCentroid</span><span class="p">(</span><span class="n">p</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
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
<pre>&lt;NearestCentroid&gt;
</pre>
</div>
</div>

</div>
</div>

</div>