<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="regressor.export_graphviz">regressor.export_graphviz<a class="anchor-link" href="#regressor.export_graphviz">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">regressor</span><span class="o">.</span><span class="n">export_graphviz</span><span class="p">(</span><span class="n">tree_id</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Converts the input tree to graphviz.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">tree_id</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Unique tree identifier. It is an integer between 0 and n_estimators - 1</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>str</b> : graphviz formatted tree.</p>

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
<div class="prompt input_prompt">In&nbsp;[18]:</div>
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
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">export_graphviz</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>digraph Tree{
1 [label = &#34;fixed_acidity &lt; 7.581250 ?&#34;, color=&#34;blue&#34;];
1 -&gt; 2 [label = &#34;yes&#34;, color = &#34;black&#34;];
1 -&gt; 3 [label = &#34;no&#34;, color = &#34;black&#34;];
2 [label = &#34;fixed_acidity &lt; 5.690625 ?&#34;, color=&#34;blue&#34;];
2 -&gt; 4 [label = &#34;yes&#34;, color = &#34;black&#34;];
2 -&gt; 5 [label = &#34;no&#34;, color = &#34;black&#34;];
4 [label = &#34;prediction: 6.056604, variance: 0.996796&#34;, color=&#34;red&#34;];
5 [label = &#34;fixed_acidity &lt; 6.068750 ?&#34;, color=&#34;blue&#34;];
5 -&gt; 10 [label = &#34;yes&#34;, color = &#34;black&#34;];
5 -&gt; 11 [label = &#34;no&#34;, color = &#34;black&#34;];
10 [label = &#34;prediction: 0.000000, variance: nan&#34;, color=&#34;red&#34;];
11 [label = &#34;prediction: 5.868771, variance: 0.740586&#34;, color=&#34;red&#34;];
3 [label = &#34;alcohol &lt; 11.018750 ?&#34;, color=&#34;blue&#34;];
3 -&gt; 6 [label = &#34;yes&#34;, color = &#34;black&#34;];
3 -&gt; 7 [label = &#34;no&#34;, color = &#34;black&#34;];
6 [label = &#34;fixed_acidity &lt; 10.984375 ?&#34;, color=&#34;blue&#34;];
6 -&gt; 12 [label = &#34;yes&#34;, color = &#34;black&#34;];
6 -&gt; 13 [label = &#34;no&#34;, color = &#34;black&#34;];
12 [label = &#34;prediction: 5.467577, variance: 0.58797&#34;, color=&#34;red&#34;];
13 [label = &#34;prediction: 5.756410, variance: 0.748356&#34;, color=&#34;red&#34;];
7 [label = &#34;alcohol &lt; 11.665625 ?&#34;, color=&#34;blue&#34;];
7 -&gt; 14 [label = &#34;yes&#34;, color = &#34;black&#34;];
7 -&gt; 15 [label = &#34;no&#34;, color = &#34;black&#34;];
14 [label = &#34;prediction: 5.918033, variance: 0.665413&#34;, color=&#34;red&#34;];
15 [label = &#34;prediction: 6.387097, variance: 0.56905&#34;, color=&#34;red&#34;];
}
</pre>
</div>
</div>

</div>
</div>

</div>