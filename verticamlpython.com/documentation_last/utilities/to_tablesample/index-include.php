<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="to_tablesample">to_tablesample<a class="anchor-link" href="#to_tablesample">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">to_tablesample</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
               <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
               <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;Sample&quot;</span><span class="p">,</span>
               <span class="n">query_on</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
               <span class="n">time_on</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
               <span class="n">title</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Returns the Result of a SQL query as a tablesample object.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">query</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>SQL Query.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Name of the object. It is used only for rendering purposes.</td> </tr>
    <tr> <td><div class="param_name">query_on</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, display the query.</td> </tr>
    <tr> <td><div class="param_name">time_on</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, displays the query elapsed time.</td> </tr>
    <tr> <td><div class="param_name">title</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Query title when the query is displayed.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><a href="../tablesample/index.php">tablesample</a> : An object containing the result. For more information, check out <a href="../tablesample/index.php">utilities.tablesample</a>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">to_tablesample</span><span class="p">(</span><span class="s2">&quot;SELECT * FROM public.titanic LIMIT 5&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Belfast, NI</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">Montevideo, Uruguay</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[1]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: Sample, Number of rows: 5, Number of columns: 14</pre>
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
    <tr><td><a href="../tablesample/index.php">tablesample</a></td> <td>Object in memory created for rendering purposes.</td></tr>
</table>
</div>
</div>
</div>