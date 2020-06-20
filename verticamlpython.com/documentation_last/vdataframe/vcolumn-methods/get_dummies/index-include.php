<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame[].get_dummies">vDataFrame[].get_dummies<a class="anchor-link" href="#vDataFrame[].get_dummies">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="p">[]</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">prefix</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span> 
                         <span class="n">prefix_sep</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;_&quot;</span><span class="p">,</span> 
                         <span class="n">drop_first</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span> 
                         <span class="n">use_numbers_as_suffix</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Encodes the vcolumn using the One Hot Encoding algorithm.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">prefix</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Prefix of the dummies.</td> </tr>
    <tr> <td><div class="param_name">prefix_sep</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Prefix delimitor of the dummies.</td> </tr>
    <tr> <td><div class="param_name">drop_first</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>Drops the first dummy to avoid the creation of correlated features.</td> </tr>
    <tr> <td><div class="param_name">use_numbers_as_suffix</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>Uses numbers as suffix instead of the vcolumns categories.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : self.parent</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataFrame</span>
<span class="n">churn</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.churn&quot;</span><span class="p">)</span>
<span class="n">churn</span> <span class="o">=</span> <span class="n">churn</span><span class="o">.</span><span class="n">select</span><span class="p">([</span><span class="s2">&quot;InternetService&quot;</span><span class="p">,</span> <span class="s2">&quot;MonthlyCharges&quot;</span><span class="p">,</span> <span class="s2">&quot;churn&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">churn</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>MonthlyCharges</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Churn</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">65.600</td><td style="border: 1px solid white;">False</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">59.900</td><td style="border: 1px solid white;">False</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">73.900</td><td style="border: 1px solid white;">True</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">98.000</td><td style="border: 1px solid white;">True</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">83.900</td><td style="border: 1px solid white;">True</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: churn, Number of rows: 7043, Number of columns: 3
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">churn</span><span class="p">[</span><span class="s2">&quot;InternetService&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>MonthlyCharges</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Churn</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService_DSL</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService_Fiber optic</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">65.600</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">59.900</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">73.900</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">98.000</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">83.900</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[28]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: churn, Number of rows: 7043, Number of columns: 5</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Number as suffix</span>
<span class="n">churn</span><span class="p">[</span><span class="s2">&quot;InternetService&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">use_numbers_as_suffix</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>MonthlyCharges</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Churn</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService_0</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService_1</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">65.600</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">59.900</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">73.900</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">98.000</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">83.900</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[30]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: churn, Number of rows: 7043, Number of columns: 5</pre>
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
    <tr><td><a href="../decode">vDataFrame[].decode</a></td> <td>Encodes the vcolumn using a user defined Encoding.</td></tr>
    <tr><td><a href="../discretize">vDataFrame[].discretize</a></td> <td>Discretizes the vcolumn.</td></tr>
    <tr><td><a href="../label_encode">vDataFrame[].label_encode</a></td> <td>Encodes the vcolumn using the Label Encoding.</td></tr>
    <tr><td><a href="../mean_encode">vDataFrame[].mean_encode</a></td> <td>Encodes the vcolumn using the Mean Encoding of a response.</td></tr>
</table>
</div>
</div>
</div>