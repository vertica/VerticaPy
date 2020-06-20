<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="tablesample.to_sql">tablesample.to_sql<a class="anchor-link" href="#tablesample.to_sql">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">tablesample</span><span class="o">.</span><span class="n">to_sql</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Generates the SQL query associated to the tablesample.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>str</b> : SQL query associated to the tablesample.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">tablesample</span><span class="p">(</span><span class="n">values</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;index&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;region&quot;</span><span class="p">,</span> <span class="s2">&quot;price&quot;</span><span class="p">,</span> <span class="s2">&quot;quality&quot;</span><span class="p">],</span>
                                <span class="s2">&quot;Banana&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;Brazil&quot;</span><span class="p">,</span> <span class="mf">2.3</span><span class="p">,</span> <span class="mi">2</span><span class="p">],</span>
                                <span class="s2">&quot;Manguo&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;Columbia&quot;</span><span class="p">,</span> <span class="mf">6.7</span><span class="p">,</span> <span class="mi">5</span><span class="p">],</span>
                                <span class="s2">&quot;Apple&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;France&quot;</span><span class="p">,</span> <span class="mf">1.5</span><span class="p">,</span> <span class="mi">2</span><span class="p">]},</span>
                      <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;Fruits&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">dataset</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Banana</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Manguo</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Apple</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>region</b></td><td style="border: 1px solid white;">Brazil</td><td style="border: 1px solid white;">Columbia</td><td style="border: 1px solid white;">France</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>price</b></td><td style="border: 1px solid white;">2.3</td><td style="border: 1px solid white;">6.7</td><td style="border: 1px solid white;">1.5</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>quality</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">2</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: Fruits, Number of rows: 3, Number of columns: 4
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">to_sql</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>(SELECT &#39;region&#39; AS &#34;index&#34;, &#39;Brazil&#39; AS &#34;Banana&#34;, &#39;Columbia&#39; AS &#34;Manguo&#34;, &#39;France&#39; AS &#34;Apple&#34;) UNION ALL (SELECT &#39;price&#39; AS &#34;index&#34;, 2.3 AS &#34;Banana&#34;, 6.7 AS &#34;Manguo&#34;, 1.5 AS &#34;Apple&#34;) UNION ALL (SELECT &#39;quality&#39; AS &#34;index&#34;, 2 AS &#34;Banana&#34;, 5 AS &#34;Manguo&#34;, 2 AS &#34;Apple&#34;)
</pre>
</div>
</div>

</div>
</div>

</div>