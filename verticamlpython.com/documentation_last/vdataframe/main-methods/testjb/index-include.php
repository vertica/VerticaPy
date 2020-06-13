<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.testjb">vDataFrame.testjb<a class="anchor-link" href="#vDataFrame.testjb">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">testjb</span><span class="p">(</span><span class="n">column</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
                  <span class="n">alpha</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.05</span><span class="p">,</span> 
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
<p>Jarque Bera test (Distribution Normality).</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">column</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Input vcolumn to test.</td> </tr>
    <tr> <td><div class="param_name">alpha</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Significance Level. Probability to accept H0.</td> </tr>
    <tr> <td><div class="param_name">print_info</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, displays all the test information.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>tuple</b> : <br> [0] result of the test : True if H0 was rejected, <br>
               [1]                 jb : Jarque Bera Index, <br>
               [2]            p_value : p value of the coefficient</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[71]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span>
<span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_titanic</span>
<span class="n">titanic</span> <span class="o">=</span> <span class="n">load_titanic</span><span class="p">()</span>
<span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[71]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 14</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[73]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">testjb</span><span class="p">(</span><span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;age&quot;</span><span class="p">,</span>
               <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.01</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>We test the following hypothesis:
(H0) The distribution of &#34;age&#34; is not normal
(H1) The distribution of &#34;age&#34; is normal
👍 - The distribution of &#34;age&#34; might be normal
jb = 28.5338631758186
p_value = 0.0
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[73]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(True, 28.5338631758186, 0.0)</pre>
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
    <tr><td><a href="../testdf">vDataFrame.testdf</a></td> <td>Dickey Fuller test.</td></tr>
    <tr><td><a href="../testdw">vDataFrame.testdw</a></td> <td>Durbin Watson test.</td></tr>
    <tr><td><a href="../testmk">vDataFrame.testmk</a></td> <td>Mann Kendall test.</td></tr>
</table>
</div>
</div>
</div>