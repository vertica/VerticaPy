<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.expected_store_usage">vDataFrame.expected_store_usage<a class="anchor-link" href="#vDataFrame.expected_store_usage">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">expected_store_usage</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Returns the vDataFrame expected store usage.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">unit</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>unit used for the computation.<br>
                                                    <ul>
                                                        <li><b>b :</b> byte</li>
                                                        <li><b>kb :</b> kilo byte</li>
                                                        <li><b>gb :</b> giga byte</li>
                                                        <li><b>tb :</b> tera byte</li></ul></td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><a href="../../../utilities/tablesample/index.php">tablesample</a> : An object containing the result. For more information, check out <a href="../../../utilities/tablesample/index.php">utilities.tablesample</a>.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Form</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Price</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Name</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">1.1193087167</td><td style="border: 1px solid white;">Acorn squash</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">1.1722478842</td><td style="border: 1px solid white;">Acorn squash</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">1.56751539145</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">1.6155336441</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.5104657455</td><td style="border: 1px solid white;">Apples</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
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
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">market</span><span class="o">.</span><span class="n">expected_store_usage</span><span class="p">(</span><span class="n">unit</span> <span class="o">=</span> <span class="s2">&quot;kb&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>expected_size (kb)</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>max_size (kb)</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>type</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"Form"</b></td><td style="border: 1px solid white;">2.423828125</td><td style="border: 1px solid white;">9.8125</td><td style="border: 1px solid white;">varchar(32)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"Price"</b></td><td style="border: 1px solid white;">0.0078125</td><td style="border: 1px solid white;">2.453125</td><td style="border: 1px solid white;">float</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"Name"</b></td><td style="border: 1px solid white;">2.888671875</td><td style="border: 1px solid white;">9.8125</td><td style="border: 1px solid white;">varchar(32)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>separator</b></td><td style="border: 1px solid white;">0.919921875</td><td style="border: 1px solid white;">0.919921875</td><td style="border: 1px solid white;"></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>header</b></td><td style="border: 1px solid white;">0.021484375</td><td style="border: 1px solid white;">0.021484375</td><td style="border: 1px solid white;"></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>rawsize</b></td><td style="border: 1px solid white;">6.26171875</td><td style="border: 1px solid white;">23.01953125</td><td style="border: 1px solid white;"></td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
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
    <tr><td><a href="../memory_usage/index.php">vDataFrame.memory_usage</a></td> <td>Returns the vDataFrame memory usage.</td></tr>
</table>
</div>
</div>
</div>