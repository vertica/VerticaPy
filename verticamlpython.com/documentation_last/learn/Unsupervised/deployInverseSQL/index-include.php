<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="unsupervised.deployInverseSQL">unsupervised.deployInverseSQL<a class="anchor-link" href="#unsupervised.deployInverseSQL">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">unsupervised</span><span class="o">.</span><span class="n">deployInverseSQL</span><span class="p">(</span><span class="n">key_columns</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Returns the SQL code needed to deploy the inverse model.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">key_columns</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>Predictors used during the algorithm computation which will be deployed with the singular values.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>str / list</b> : the SQL code needed to deploy the inverse model.</p>

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
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.decomposition</span> <span class="k">import</span> <span class="n">PCA</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">PCA</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.pca_iris&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalWidthCm&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">deployInverseSQL</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>APPLY_INVERSE_PCA(&#34;PetalLengthCm&#34;, &#34;SepalLengthCm&#34;, &#34;SepalWidthCm&#34; USING PARAMETERS model_name = &#39;public.pca_iris&#39;, match_by_pos = &#39;true&#39;)
</pre>
</div>
</div>

</div>
</div>

</div>