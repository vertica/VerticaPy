<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="to_vertica_python_format">to_vertica_python_format<a class="anchor-link" href="#to_vertica_python_format">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">to_vertica_python_format</span><span class="p">(</span><span class="n">dsn</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Converts the ODBC dictionary obtained with the read_dsn method to the vertica_python format.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">dsn</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>DSN name</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>dict</b> : dictionary with all the credentials</p>

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
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.connections.connect</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">to_vertica_python_format</span><span class="p">(</span><span class="s2">&quot;VerticaDSN&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[17]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;database&#39;: &#39;testdb&#39;,
 &#39;host&#39;: &#39;10.211.55.14&#39;,
 &#39;password&#39;: &#39;XxX&#39;,
 &#39;port&#39;: 5433,
 &#39;user&#39;: &#39;dbadmin&#39;}</pre>
</div>

</div>

</div>
</div>

</div>