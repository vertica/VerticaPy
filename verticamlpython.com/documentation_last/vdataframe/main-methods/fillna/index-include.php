<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.fillna">vDataFrame.fillna<a class="anchor-link" href="#vDataFrame.fillna">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">val</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{},</span>
                  <span class="n">method</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{},</span>
                  <span class="n">numeric_only</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
                  <span class="n">print_info</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Fills the vcolumns missing elements using specific rules.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">val</div></td> <td><div class="type">dict</div></td> <td><div class = "yes">&#10003;</div></td> <td>Dictionary of values. The dictionary must be similar to the following: {"column1": val1 ..., "columnk": valk}. Each key of the dictionary must be a vcolumn. The missing values of the input vcolumns will be replaced by the input value.</td> </tr>
    <tr> <td><div class="param_name">method</div></td> <td><div class="type">dict</div></td> <td><div class = "yes">&#10003;</div></td> <td>Method used to impute the missing values.<br>
                                                    <ul>
                                                        <li><b>auto :</b> Mean for the numerical and Mode for the categorical vcolumns.</li>
                                                        <li><b>mean :</b> Average.</li>
                                                        <li><b>median :</b> Median.</li>
                                                        <li><b>mode :</b> Mode (most occurent element).</li>
                                                        <li><b>0ifnull :</b> 0 when the vcolumn is null, 1 otherwise.</li></ul>More Methods are available on the vDataFrame[].fillna method.</td> </tr>
    <tr> <td><div class="param_name">numeric_only</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If parameters 'val' and 'method' are empty and 'numeric_only' is set to True then all the numerical vcolumns will be imputed by their average. If set to False, all the categorical vcolumns will be also imputed by their mode.</td> </tr>
    <tr> <td><div class="param_name">print_info</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, displays all the filling information.</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[53]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_titanic</span>
<span class="n">titanic</span> <span class="o">=</span> <span class="n">load_titanic</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">titanic</span><span class="p">)</span>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 14
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>percent</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"ticket"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"name"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">1233.0</td><td style="border: 1px solid white;">99.919</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"embarked"</b></td><td style="border: 1px solid white;">1232.0</td><td style="border: 1px solid white;">99.838</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"age"</b></td><td style="border: 1px solid white;">997.0</td><td style="border: 1px solid white;">80.794</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"home.dest"</b></td><td style="border: 1px solid white;">706.0</td><td style="border: 1px solid white;">57.212</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">439.0</td><td style="border: 1px solid white;">35.575</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"cabin"</b></td><td style="border: 1px solid white;">286.0</td><td style="border: 1px solid white;">23.177</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"body"</b></td><td style="border: 1px solid white;">118.0</td><td style="border: 1px solid white;">9.562</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[53]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">val</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;boat&quot;</span><span class="p">:</span> <span class="s2">&quot;No boat&quot;</span><span class="p">},</span>
               <span class="n">method</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;age&quot;</span><span class="p">:</span> <span class="s2">&quot;mean&quot;</span><span class="p">,</span>
                         <span class="s2">&quot;embarked&quot;</span><span class="p">:</span> <span class="s2">&quot;mode&quot;</span><span class="p">,</span>
                         <span class="s2">&quot;fare&quot;</span><span class="p">:</span> <span class="s2">&quot;median&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">No boat</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0000000000000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">No boat</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.0000000000000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">No boat</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.0000000000000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">No boat</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.0000000000000</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">No boat</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.0000000000000</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[54]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 14</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[55]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>percent</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"ticket"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"embarked"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"age"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"name"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">100.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"home.dest"</b></td><td style="border: 1px solid white;">706.0</td><td style="border: 1px solid white;">57.212</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"cabin"</b></td><td style="border: 1px solid white;">286.0</td><td style="border: 1px solid white;">23.177</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"body"</b></td><td style="border: 1px solid white;">118.0</td><td style="border: 1px solid white;">9.562</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[55]:</div>




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
    <tr><td><a href="../../vcolumn-methods/fillna">vDataFrame[].fillna</a></td> <td>Fills the vcolumn missing values. This method is more complete than the vDataFrame.fillna method by allowing more parameters.</td></tr>
</table>
</div>
</div>
</div>