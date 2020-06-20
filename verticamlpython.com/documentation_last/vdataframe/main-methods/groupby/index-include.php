<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.groupby">vDataFrame.groupby<a class="anchor-link" href="#vDataFrame.groupby">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">columns</span><span class="p">:</span> <span class="nb">list</span><span class="p">,</span> 
                   <span class="n">expr</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Aggregates the vDataFrame by grouping the elements.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">columns</div></td> <td><div class="type">list</div></td> <td><div class = "no">&#10060;</div></td> <td>List of the vcolumns used for the grouping.</td> </tr>
    <tr> <td><div class="param_name">expr</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the different aggregations. Pure SQL must be written. Aliases can also be given. 'SUM(column)' or 'AVG(column) AS my_new_alias' are correct whereas 'AVG' is incorrect. Aliases are recommended to keep the track of the different features and not have ambiguous names. The function MODE does not exist in SQL for example but can be obtained using the 'analytic' method first and then by grouping the result.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : object result of the grouping.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_market</span>
<span class="n">market</span> <span class="o">=</span> <span class="n">load_market</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">market</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Form</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Price</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Name</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">1.1193087167</td><td style="border: 1px solid white;">Acorn squash</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">1.1722478842</td><td style="border: 1px solid white;">Acorn squash</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">1.56751539145</td><td style="border: 1px solid white;">Apples</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">1.6155336441</td><td style="border: 1px solid white;">Apples</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.5104657455</td><td style="border: 1px solid white;">Apples</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: market, Number of rows: 314, Number of columns: 3
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">market</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Form&quot;</span><span class="p">,</span> <span class="s2">&quot;Name&quot;</span><span class="p">],</span>
               <span class="n">expr</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;AVG(Price) AS avg_price&quot;</span><span class="p">,</span>
                       <span class="s2">&quot;STDDEV(Price) AS std&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Form</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_price</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>std</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">Acorn squash</td><td style="border: 1px solid white;">1.14577830045</td><td style="border: 1px solid white;">0.0374336443296234</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">Apples</td><td style="border: 1px solid white;">1.591524517775</td><td style="border: 1px solid white;">0.033954032069563</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">Apples</td><td style="border: 1px solid white;">0.5241668305185</td><td style="border: 1px solid white;">0.0193762602523861</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Ready to drink</td><td style="border: 1px solid white;">Apples</td><td style="border: 1px solid white;">0.6792101204525</td><td style="border: 1px solid white;">0.0679919835754149</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Dried</td><td style="border: 1px solid white;">Apricots</td><td style="border: 1px solid white;">7.532468311145</td><td style="border: 1px solid white;">0.284969376912029</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[37]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 159, Number of columns: 4</pre>
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
    <tr><td><a href="../append/index.php">vDataFrame.append</a></td> <td>Merges the vDataFrame with another relation.</td></tr>
    <tr><td><a href="../analytic/index.php">vDataFrame.analytic</a></td> <td>Adds a new vcolumn to the vDataFrame by using an advanced analytical function on a specific vcolumn.</td></tr>
    <tr><td><a href="../join/index.php">vDataFrame.join</a></td> <td>Joins the vDataFrame with another relation.</td></tr>
    <tr><td><a href="../sort/index.php">vDataFrame.sort</a></td> <td>Sorts the vDataFrame.</td></tr>
</table>
</div>
</div>
</div>