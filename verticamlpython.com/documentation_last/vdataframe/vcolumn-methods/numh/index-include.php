<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame[].numh">vDataFrame[].numh<a class="anchor-link" href="#vDataFrame[].numh">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="p">[]</span><span class="o">.</span><span class="n">numh</span><span class="p">(</span><span class="n">method</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes the optimal vcolumn bar width.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">method</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Method used to compute the optimal h.<br>
                                                    <ul>
                                                        <li><b>auto :</b> Combination of Freedman Diaconis and Sturges.</li>
                                                        <li><b>freedman_diaconis :</b> Freedman Diaconis [2 * IQR / n ** (1 / 3)]</li>
                                                        <li><b>sturges :</b> Sturges [CEIL(log2(n)) + 1]</li>
                                                        </ul></td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>float</b> : optimal bar width.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
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
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">market</span><span class="p">[</span><span class="s2">&quot;Price&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">numh</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;freedman_diaconis&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0.45050173897623425</pre>
</div>

</div>

</div>
</div>

</div>