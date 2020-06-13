<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.cumprod">vDataFrame.cumprod<a class="anchor-link" href="#vDataFrame.cumprod">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">cumprod</span><span class="p">(</span><span class="n">column</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
                   <span class="n">by</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span> 
                   <span class="n">order_by</span> <span class="o">=</span> <span class="p">[],</span>
                   <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Adds a new vcolumn to the vDataFrame by computing the cumulative product of the input vcolumn.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">column</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Input vcolumn.</td> </tr>
    <tr> <td><div class="param_name">by</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>vcolumns used in the partition.</td> </tr>
    <tr> <td><div class="param_name">order_by</div></td> <td><div class="type">dict / list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the vcolumns used to sort the data using asc order or dictionary of all the sorting methods. For example, to sort by "column1" ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}</td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Name of the new vcolumn. If empty, a default name will be generated.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : self</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[53]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_amazon</span>
<span class="n">amazon</span> <span class="o">=</span> <span class="n">load_amazon</span><span class="p">()</span><span class="o">.</span><span class="n">last</span><span class="p">(</span><span class="n">ts</span> <span class="o">=</span> <span class="s2">&quot;date&quot;</span><span class="p">,</span> <span class="n">offset</span> <span class="o">=</span> <span class="s2">&quot;6 months&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">amazon</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>6292 element(s) was/were filtered
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>date</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>number</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>state</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">2017-06-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Paraiba</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">2017-06-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Sergipe</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">2017-07-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Alagoas</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">2017-07-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Sergipe</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">2017-08-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Sergipe</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: amazon, Number of rows: 162, Number of columns: 3
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[54]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">amazon</span><span class="o">.</span><span class="n">cumprod</span><span class="p">(</span><span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;number&quot;</span><span class="p">,</span>
               <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;state&quot;</span><span class="p">],</span>
               <span class="n">order_by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;date&quot;</span><span class="p">],</span>
               <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;cumprod_number&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>date</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>number</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>state</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cumprod_number</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">2017-06-01</td><td style="border: 1px solid white;">45</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">45.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">2017-07-01</td><td style="border: 1px solid white;">457</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">20565.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">2017-08-01</td><td style="border: 1px solid white;">1493</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">30703545.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">2017-09-01</td><td style="border: 1px solid white;">3429</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">105282455805.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">2017-10-01</td><td style="border: 1px solid white;">1508</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">158765943353940.0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[54]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: amazon, Number of rows: 162, Number of columns: 4</pre>
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
    <tr><td><a href="../rolling/index.php">vDataFrame.rolling</a></td> <td> Computes a customized moving window.</td></tr>
</table>
</div>
</div>
</div>