<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="regressor.features_importance">regressor.features_importance<a class="anchor-link" href="#regressor.features_importance">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">regressor</span><span class="o">.</span><span class="n">features_importance</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes the model features importance.</p>
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><a href="../../../utilities/tablesample/index.php">tablesample</a> : An object containing the result. For more information, check out <a href="../../../utilities/tablesample/index.php">utilities.tablesample</a>.</p>

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
<div class="prompt input_prompt">In&nbsp;[31]:</div>
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
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.winequality&quot;</span><span class="p">,</span> 
          <span class="p">[</span><span class="s2">&quot;alcohol&quot;</span><span class="p">,</span> <span class="s2">&quot;fixed_acidity&quot;</span><span class="p">,</span> <span class="s2">&quot;pH&quot;</span><span class="p">,</span> <span class="s2">&quot;density&quot;</span><span class="p">],</span> 
          <span class="s2">&quot;quality&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">features_importance</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>importance</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>alcohol</b></td><td style="border: 1px solid white;">71.71</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>density</b></td><td style="border: 1px solid white;">24.34</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>ph</b></td><td style="border: 1px solid white;">2.24</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>fixed_acidity</b></td><td style="border: 1px solid white;">1.71</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[31]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>