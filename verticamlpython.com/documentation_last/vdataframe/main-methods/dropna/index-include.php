<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.dropna">vDataFrame.dropna<a class="anchor-link" href="#vDataFrame.dropna">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">dropna</span><span class="p">(</span><span class="n">columns</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span> 
                  <span class="n">print_info</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Filters the vDataFrame where the input vcolumns are missing.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">columns</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the vcolumns names. If empty, all the vcolumns will be selected.</td> </tr>
    <tr> <td><div class="param_name">print_info</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, it will display the result.</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_titanic</span>
<span class="n">titanic</span> <span class="o">=</span> <span class="n">load_titanic</span><span class="p">()</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>percent</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"ticket"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"name"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">1233.0</td><td style="border: 1px solid white;">99.919</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"embarked"</b></td><td style="border: 1px solid white;">1232.0</td><td style="border: 1px solid white;">99.838</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"age"</b></td><td style="border: 1px solid white;">997.0</td><td style="border: 1px solid white;">80.794</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"home.dest"</b></td><td style="border: 1px solid white;">706.0</td><td style="border: 1px solid white;">57.212</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">439.0</td><td style="border: 1px solid white;">35.575</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"cabin"</b></td><td style="border: 1px solid white;">286.0</td><td style="border: 1px solid white;">23.177</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"body"</b></td><td style="border: 1px solid white;">118.0</td><td style="border: 1px solid white;">9.562</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[40]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">dropna</span><span class="p">(</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;fare&quot;</span><span class="p">,</span> <span class="s2">&quot;embarked&quot;</span><span class="p">,</span> <span class="s2">&quot;age&quot;</span><span class="p">])</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>240 element(s) was/were dropped
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>percent</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"ticket"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"embarked"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"age"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"name"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">994.0</td><td style="border: 1px solid white;">100.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"home.dest"</b></td><td style="border: 1px solid white;">648.0</td><td style="border: 1px solid white;">65.191</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">380.0</td><td style="border: 1px solid white;">38.229</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"cabin"</b></td><td style="border: 1px solid white;">261.0</td><td style="border: 1px solid white;">26.258</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"body"</b></td><td style="border: 1px solid white;">116.0</td><td style="border: 1px solid white;">11.67</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[41]:</div>




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
    <tr><td><a href="../filter">vDataFrame.filter</a></td> <td>Filters the data using the input expression.</td></tr>
</table>
</div>
</div>
</div>