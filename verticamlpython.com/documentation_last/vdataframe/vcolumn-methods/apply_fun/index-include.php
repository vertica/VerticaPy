<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame[].apply_fun">vDataFrame[].apply_fun<a class="anchor-link" href="#vDataFrame[].apply_fun">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="p">[]</span><span class="o">.</span><span class="n">apply_fun</span><span class="p">(</span><span class="n">func</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                       <span class="n">x</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Applies a default function to the vcolumn.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">func</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Function used to transform the vcolumn.<br>
                                                    <ul>
                                                        <li><b>abs :</b> absolute value</li>
                                                        <li><b>acos :</b> trigonometric inverse cosine</li>
                                                        <li><b>asin :</b> trigonometric inverse sine</li>
                                                        <li><b>atan :</b> trigonometric inverse tangent</li>
                                                        <li><b>cbrt :</b> cube root</li>
                                                        <li><b>ceil :</b> value up to the next whole number</li>
                                                        <li><b>cos :</b> trigonometric cosine</li>
                                                        <li><b>cosh :</b> hyperbolic cosine</li>
                                                        <li><b>cot:</b> trigonometric cotangent</li>
                                                        <li><b>exp :</b> exponential function</li>
                                                        <li><b>floor :</b> value down to the next whole number</li>
                                                        <li><b>ln :</b> natural logarithm</li>
                                                        <li><b>log :</b> logarithm</li>
                                                        <li><b>log10 :</b> base 10 logarithm</li>
                                                        <li><b>mod :</b> remainder of a division operation</li>
                                                        <li><b>pow :</b> number raised to the power of another number</li>
                                                        <li><b>round :</b> rounds a value to a specified number of decimal places</li>
                                                        <li><b>sign :</b> arithmetic sign</li>
                                                        <li><b>sin :</b> trigonometric sine</li>
                                                        <li><b>sinh :</b> hyperbolic sine</li>
                                                        <li><b>sqrt :</b> arithmetic square root</li>
                                                        <li><b>tan :</b> trigonometric tangent</li>
                                                        <li><b>tanh :</b> hyperbolic tangent</li></ul></td> </tr>
    <tr> <td><div class="param_name">x</div></td> <td><div class="type">int / float</div></td> <td><div class = "yes">&#10003;</div></td> <td>If the function has two arguments (example, power or mod), 'x' represents the second argument.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : self.parent</p>
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
<span class="nb">print</span><span class="p">(</span><span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">2.000</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">30.000</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">25.000</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">39.000</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">71.000</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: age, Number of rows: 1234, dtype: numeric(6,3)
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">apply_fun</span><span class="p">(</span><span class="n">func</span> <span class="o">=</span> <span class="s2">&quot;log10&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0.301029995663981</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1.47712125471966</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1.39794000867204</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">1.5910646070265</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">1.85125834871908</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[42]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: age, Number of rows: 1234, dtype: float</pre>
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
    <tr><td><a href="../apply/index.php">vDataFrame[].apply</a></td> <td> Applies a function to the vcolumn.</td></tr>
</table>
</div>
</div>
</div>