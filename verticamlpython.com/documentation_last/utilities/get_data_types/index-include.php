<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="get_data_types">get_data_types<a class="anchor-link" href="#get_data_types">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">get_data_types</span><span class="p">(</span><span class="n">table</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
               <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
               <span class="n">column_name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span> 
               <span class="n">schema_writing</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Returns a customized relation columns and the respective data types. It will create a temporary table during the process.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">table</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Relation. It must be pure SQL.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">column_name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>If not empty, it will return only the data type of the input column if it is in the relation.</td> </tr>
    <tr> <td><div class="param_name">schema_writing</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Schema used to create the temporary table. If empty, the function will create a local temporary table.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>list of tuples</b> : The list of the different columns and their respective type.</p>

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
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">get_data_types</span><span class="p">(</span><span class="s2">&quot;SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>[[&#39;pclass&#39;, &#39;int&#39;], [&#39;embarked&#39;, &#39;varchar(20)&#39;], [&#39;AVG&#39;, &#39;float&#39;]]</pre>
</div>

</div>

</div>
</div>

</div>