<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="regressor.deploySQL">regressor.deploySQL<a class="anchor-link" href="#regressor.deploySQL">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">regressor</span><span class="o">.</span><span class="n">deploySQL</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Returns the SQL code needed to deploy the model.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>str</b> : the SQL code needed to deploy the model.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestRegressor</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestRegressor</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.RF_winequality&quot;</span><span class="p">,</span>
                              <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span>
                              <span class="n">max_features</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                              <span class="n">max_leaf_nodes</span> <span class="o">=</span> <span class="mi">32</span><span class="p">,</span> 
                              <span class="n">sample</span> <span class="o">=</span> <span class="mf">0.7</span><span class="p">,</span>
                              <span class="n">max_depth</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span>
                              <span class="n">min_samples_leaf</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                              <span class="n">min_info_gain</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                              <span class="n">nbins</span> <span class="o">=</span> <span class="mi">32</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.winequality&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;alcohol&quot;</span><span class="p">,</span> <span class="s2">&quot;fixed_acidity&quot;</span><span class="p">],</span> <span class="s2">&quot;quality&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">deploySQL</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>PREDICT_RF_REGRESSOR(&#34;alcohol&#34;, &#34;fixed_acidity&#34; USING PARAMETERS model_name = &#39;public.RF_winequality&#39;, match_by_pos = &#39;true&#39;)
</pre>
</div>
</div>

</div>
</div>

</div>