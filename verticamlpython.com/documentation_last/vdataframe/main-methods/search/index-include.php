<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.search">vDataFrame.search<a class="anchor-link" href="#vDataFrame.search">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="n">conditions</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
                  <span class="n">usecols</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                  <span class="n">expr</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                  <span class="n">order_by</span> <span class="o">=</span> <span class="p">[])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Searches the elements which matches with the input conditions.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">conditions</div></td> <td><div class="type">str / list</div></td> <td><div class = "yes">&#10003;</div></td> <td>Filters of the search. It can be a list of conditions or an expression.</td> </tr>
    <tr> <td><div class="param_name">usecols</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>vcolumns to select from the final vDataFrame relation. If empty, all the vcolumns will be selected.</td> </tr>
    <tr> <td><div class="param_name">expr</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of customized expressions. It must be pure SQL. For example, it is possible to write 'column1 * column2 AS my_name'.</td> </tr>
    <tr> <td><div class="param_name">order_by</div></td> <td><div class="type">dict / list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the vcolumns used to sort the data using asc order or dictionary of all the sorting methods. For example, to sort by "column1" ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : vDataFrame of the search</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_titanic</span>
<span class="n">titanic</span> <span class="o">=</span> <span class="n">load_titanic</span><span class="p">()</span>
<span class="c1"># Looking at the family size and survival of the passengers having</span>
<span class="c1"># more than 50 years old who paid the most expensive fares</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="n">conditions</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;age &gt; 50&quot;</span><span class="p">],</span>
               <span class="n">usecols</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;fare&quot;</span><span class="p">,</span> <span class="s2">&quot;survived&quot;</span><span class="p">],</span>
               <span class="n">expr</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;parch + sibsp + 1 AS family_size&quot;</span><span class="p">],</span>
               <span class="n">order_by</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;fare&quot;</span><span class="p">:</span> <span class="s2">&quot;desc&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>family_size</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">512.32920</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">263.00000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">6</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">263.00000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">6</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">262.37500</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">5</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">221.77920</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[1]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: search, Number of rows: 95, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="See-Also">See Also<a class="anchor-link" href="#See-Also">&#182;</a></h3><table id="seealso">
    <tr><td><a href="../filter">vDataFrame.filter</a></td> <td>Filters the vDataFrame using the input expressions.</td></tr>
</table>
</div>
</div>
</div>