<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="classification_report">classification_report<a class="anchor-link" href="#classification_report">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_true</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span> 
                      <span class="n">y_score</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span> 
                      <span class="n">input_relation</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
                      <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
                      <span class="n">labels</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                      <span class="n">cutoff</span> <span class="o">=</span> <span class="p">[],</span>
                      <span class="n">estimator</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes a classification report using multiple metrics (AUC, accuracy, PRC AUC, F1...). It will consider each category as positive and switch to the next one during the computation.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">y_true</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Response column.</td> </tr>
    <tr> <td><div class="param_name">y_score</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List containing the probability and the prediction.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Relation used to do the scoring. The relation can be a view or a table or even a customized relation. For example, you could write: "(SELECT ... FROM ...) x" as long as an alias is given at the end of the relation.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">labels</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the response column categories to use.</td> </tr>
    <tr> <td><div class="param_name">cutoff</div></td> <td><div class="type">float / list</div></td> <td><div class = "yes">&#10003;</div></td> <td>Cutoff for which the tested category will be accepted as prediction. In case of multiclass classification, the list will represent the the classes threshold. If it is empty, the best cutoff will be used.</td> </tr>
    <tr> <td><div class="param_name">estimator</div></td> <td><div class="type">object</div></td> <td><div class = "yes">&#10003;</div></td> <td>Estimator used to compute the classification report.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><a href="../../../utilities/tablesample/index.php">tablesample</a> : An object containing the result. For more information, check out <a href="../../../utilities/tablesample/index.php">utilities.tablesample</a>.</p>

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
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataFrame</span>
<span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;example_classification&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>y_score</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>y_pred</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>y_true</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">0.261992636494471</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0.271766949212011</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0.281715565124816</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0.287704603820825</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0.293503745589547</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: example_classification, Number of rows: 1234, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.metrics</span> <span class="k">import</span> <span class="n">classification_report</span>
<span class="n">classification_report</span><span class="p">(</span><span class="s2">&quot;y_true&quot;</span><span class="p">,</span> 
                      <span class="p">[</span><span class="s2">&quot;y_score&quot;</span><span class="p">,</span> <span class="s2">&quot;y_pred&quot;</span><span class="p">],</span> 
                      <span class="s2">&quot;example_classification&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>value</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>auc</b></td><td style="border: 1px solid white;">0.6974762740166146</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>prc_auc</b></td><td style="border: 1px solid white;">0.6003540469187277</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>accuracy</b></td><td style="border: 1px solid white;">0.5996758508914101</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>log_loss</b></td><td style="border: 1px solid white;">0.281741002875517</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>precision</b></td><td style="border: 1px solid white;">0.5688888888888889</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>recall</b></td><td style="border: 1px solid white;">0.460431654676259</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>f1_score</b></td><td style="border: 1px solid white;">0.5598004843315141</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>mcc</b></td><td style="border: 1px solid white;">0.18016701318842082</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>informedness</b></td><td style="border: 1px solid white;">0.17429596146091964</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>markedness</b></td><td style="border: 1px solid white;">0.18623582766439917</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>csi</b></td><td style="border: 1px solid white;">0.3413333333333333</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>cutoff</b></td><td style="border: 1px solid white;">0.999</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[26]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>