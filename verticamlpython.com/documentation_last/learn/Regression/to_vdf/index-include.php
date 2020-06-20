<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="regressor.to_vdf">regressor.to_vdf<a class="anchor-link" href="#regressor.to_vdf">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">regressor</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Returns the vDataFrame of the Prediction.</p>
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : the vDataFrame of the prediction</p>

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
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.neighbors</span> <span class="k">import</span> <span class="n">KNeighborsRegressor</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">KNeighborsRegressor</span><span class="p">()</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.winequality&quot;</span><span class="p">,</span> 
          <span class="p">[</span><span class="s2">&quot;alcohol&quot;</span><span class="p">,</span> <span class="s2">&quot;fixed_acidity&quot;</span><span class="p">],</span> 
          <span class="s2">&quot;quality&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>alcohol</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fixed_acidity</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>quality</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">9.2</td><td style="border: 1px solid white;">6.200</td><td style="border: 1px solid white;">5.6</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">12.1</td><td style="border: 1px solid white;">6.300</td><td style="border: 1px solid white;">5.4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">10.3</td><td style="border: 1px solid white;">6.400</td><td style="border: 1px solid white;">6.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">11.7</td><td style="border: 1px solid white;">9.100</td><td style="border: 1px solid white;">6.6</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">10.6</td><td style="border: 1px solid white;">6.400</td><td style="border: 1px solid white;">5.6</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[46]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: kneighborsregressor, Number of rows: 6497, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>