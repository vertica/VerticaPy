<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="unsupervised.fit">unsupervised.fit<a class="anchor-link" href="#unsupervised.fit">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">unsupervised</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">input_relation</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                 <span class="n">X</span><span class="p">:</span> <span class="nb">list</span><span class="p">,</span>
                 <span class="n">key_columns</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span> 
                 <span class="n">index</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Trains the model.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td><div class = "no">&#10060;</div></td> <td>List of the predictors.</td> </tr>
    <tr> <td><div class="param_name">key_columns</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>Columns not used during the algorithm computation but which will be used to create the final relation.</td> </tr>
    <tr> <td><div class="param_name">index</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Index used to identify each row separately. It is highly recommanded to have one already in the main table to avoid creation of temporary tables.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>object</b> : self</p>

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
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.cluster</span> <span class="k">import</span> <span class="n">DBSCAN</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">DBSCAN</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.DBSCAN_iris&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;DBSCAN&gt;
Number of Clusters: 4
Number of Outliers: 4</pre>
</div>

</div>

</div>
</div>

</div>