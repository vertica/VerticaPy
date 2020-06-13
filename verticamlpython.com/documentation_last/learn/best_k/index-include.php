<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="best_k">best_k<a class="anchor-link" href="#best_k">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">best_k</span><span class="p">(</span><span class="n">X</span><span class="p">:</span> <span class="nb">list</span><span class="p">,</span>
       <span class="n">input_relation</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
       <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
       <span class="n">n_cluster</span> <span class="o">=</span> <span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">100</span><span class="p">),</span>
       <span class="n">init</span> <span class="o">=</span> <span class="s2">&quot;kmeanspp&quot;</span><span class="p">,</span>
       <span class="n">max_iter</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">50</span><span class="p">,</span>
       <span class="n">tol</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1e-4</span><span class="p">,</span>
       <span class="n">elbow_score_stop</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Finds the KMeans K based on a score.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td><div class = "no">&#10060;</div></td> <td>List of the predictor columns.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Relation used to train the model.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">n_cluster</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Tuple representing the number of cluster to start with and to end with. It can also be customized list with the different K to test.</td> </tr>
    <tr> <td><div class="param_name">init</div></td> <td><div class="type">str / list</div></td> <td><div class = "yes">&#10003;</div></td> <td>The method used to find the initial cluster centers. <br> <ul><li><b>kmeanspp</b> : Uses the KMeans++ method to initialize the centers.</li><li><b>random</b> : The initial centers.</li></ul>It can be also a list with the initial cluster centers to use.</td> </tr>
    <tr> <td><div class="param_name">max_iter</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The maximum number of iterations the algorithm performs.</td> </tr>
    <tr> <td><div class="param_name">tol</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Determines whether the algorithm has converged. The algorithm is considered converged after no center has moved more than a distance of 'tol' from the previous iteration. </td> </tr>
    <tr> <td><div class="param_name">elbow_score_stop</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Stops the Parameters Search when this Elbow score is reached.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>int</b> : the KMeans K.</p>

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
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.model_selection</span> <span class="k">import</span> <span class="n">best_k</span>
<span class="n">best_k</span><span class="p">(</span><span class="n">X</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">],</span>
       <span class="n">input_relation</span> <span class="o">=</span> <span class="s2">&quot;public.iris&quot;</span><span class="p">,</span>
       <span class="n">elbow_score_stop</span> <span class="o">=</span> <span class="mf">0.9</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>3</pre>
</div>

</div>

</div>
</div>

</div>