<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.at_time">vDataFrame.at_time<a class="anchor-link" href="#vDataFrame.at_time">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">at_time</span><span class="p">(</span><span class="n">ts</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                   <span class="n">time</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Filters the vDataFrame by only keeping the records at the input time.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">ts</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>TS (Time Series) vcolumn used to filter the data. The vcolumn type must be date like (date, datetime, timestamp...)</td> </tr>
    <tr> <td><div class="param_name">time</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Input Time. For example, time = '12:00' will filter the data when time('ts') is equal to 12:00.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : self</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_smart_meters</span>
<span class="n">sm</span> <span class="o">=</span> <span class="n">load_smart_meters</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">sm</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>id</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>val</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>time</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0.0370000</td><td style="border: 1px solid white;">2014-01-01 01:15:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">0.0800000</td><td style="border: 1px solid white;">2014-01-01 02:30:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0810000</td><td style="border: 1px solid white;">2014-01-01 03:00:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1.4890000</td><td style="border: 1px solid white;">2014-01-01 05:00:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">0.0720000</td><td style="border: 1px solid white;">2014-01-01 06:00:00</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: smart_meters, Number of rows: 11844, Number of columns: 3
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sm</span><span class="o">.</span><span class="n">at_time</span><span class="p">(</span><span class="n">ts</span> <span class="o">=</span> <span class="s2">&quot;time&quot;</span><span class="p">,</span> <span class="n">time</span> <span class="o">=</span> <span class="s2">&quot;12:00&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>11704 element(s) was/were filtered
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>id</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>val</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>time</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0.0760000</td><td style="border: 1px solid white;">2014-01-04 12:00:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">6</td><td style="border: 1px solid white;">0.1610000</td><td style="border: 1px solid white;">2014-01-09 12:00:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">9</td><td style="border: 1px solid white;">0.0490000</td><td style="border: 1px solid white;">2014-01-15 12:00:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0.0200000</td><td style="border: 1px solid white;">2014-01-19 12:00:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">7</td><td style="border: 1px solid white;">0.1440000</td><td style="border: 1px solid white;">2014-01-21 12:00:00</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[18]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: smart_meters, Number of rows: 140, Number of columns: 3</pre>
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
    <tr><td><a href="../between_time">vDataFrame.between_time</a></td> <td>Filters the data between two time ranges.</td></tr>
    <tr><td><a href="../first">vDataFrame.first</a></td> <td>Filters the data by only keeping the first records.</td></tr>
    <tr><td><a href="../filter">vDataFrame.filter</a></td> <td>Filters the data using the input expression.</td></tr>
    <tr><td><a href="../last">vDataFrame.last</a></td> <td>Filters the data by only keeping the last records.</td></tr>
</table>
</div>
</div>
</div>