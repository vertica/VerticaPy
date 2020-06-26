<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="tablesample.to_vdf">tablesample.to_vdf<a class="anchor-link" href="#tablesample.to_vdf">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">tablesample</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
                   <span class="n">dsn</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Converts the tablesample to a vDataFrame.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">dsn</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Data Base DSN.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : vDataFrame of the tablesample.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">verticapy.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">tablesample</span><span class="p">(</span><span class="n">values</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;index&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;region&quot;</span><span class="p">,</span> <span class="s2">&quot;price&quot;</span><span class="p">,</span> <span class="s2">&quot;quality&quot;</span><span class="p">],</span>
                                <span class="s2">&quot;Banana&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;Brazil&quot;</span><span class="p">,</span> <span class="mf">2.3</span><span class="p">,</span> <span class="mi">2</span><span class="p">],</span>
                                <span class="s2">&quot;Manguo&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;Columbia&quot;</span><span class="p">,</span> <span class="mf">6.7</span><span class="p">,</span> <span class="mi">5</span><span class="p">],</span>
                                <span class="s2">&quot;Apple&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;France&quot;</span><span class="p">,</span> <span class="mf">1.5</span><span class="p">,</span> <span class="mi">2</span><span class="p">]},</span>
                      <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;Fruits&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">transpose</span><span class="p">()</span>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>region</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>price</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>quality</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Banana</b></td><td style="border: 1px solid white;">Brazil</td><td style="border: 1px solid white;">2.3</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Manguo</b></td><td style="border: 1px solid white;">Columbia</td><td style="border: 1px solid white;">6.7</td><td style="border: 1px solid white;">5</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Apple</b></td><td style="border: 1px solid white;">France</td><td style="border: 1px solid white;">1.5</td><td style="border: 1px solid white;">2</td></tr></table>
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
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>region</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>price</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>quality</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Banana</b></td><td style="border: 1px solid white;">Brazil</td><td style="border: 1px solid white;">2.3</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Manguo</b></td><td style="border: 1px solid white;">Columbia</td><td style="border: 1px solid white;">6.7</td><td style="border: 1px solid white;">5</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Apple</b></td><td style="border: 1px solid white;">France</td><td style="border: 1px solid white;">1.5</td><td style="border: 1px solid white;">2</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: vdf, Number of rows: 3, Number of columns: 4
</pre>
</div>
</div>

</div>
</div>

</div>