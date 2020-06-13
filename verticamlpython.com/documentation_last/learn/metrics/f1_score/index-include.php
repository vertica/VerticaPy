<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="f1_score">f1_score<a class="anchor-link" href="#f1_score">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">f1_score</span><span class="p">(</span><span class="n">y_true</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
         <span class="n">y_score</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
         <span class="n">input_relation</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
         <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
         <span class="n">pos_label</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes the F1 Score.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">y_true</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Response column.</td> </tr>
    <tr> <td><div class="param_name">y_score</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Prediction.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Relation used to do the scoring. The relation can be a view or a table or even a customized relation. For example, you could write: "(SELECT ... FROM ...) x" as long as an alias is given at the end of the relation.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">pos_label</div></td> <td><div class="type">int / float / str</div></td> <td><div class = "yes">&#10003;</div></td> <td>To compute the F1 Score, one of the response column class has to be the positive one. The parameter 'pos_label' represents this class.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>float</b> : score</p>

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

<div class="prompt output_prompt">Out[3]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: example_classification, Number of rows: 1234, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.metrics</span> <span class="k">import</span> <span class="n">f1_score</span>
<span class="n">f1_score</span><span class="p">(</span><span class="s2">&quot;y_true&quot;</span><span class="p">,</span> <span class="s2">&quot;y_pred&quot;</span><span class="p">,</span> <span class="s2">&quot;example_classification&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0.5089463220675945</pre>
</div>

</div>

</div>
</div>

</div>