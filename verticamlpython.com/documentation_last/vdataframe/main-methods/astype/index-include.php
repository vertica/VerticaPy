<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.astype">vDataFrame.astype<a class="anchor-link" href="#vDataFrame.astype">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">dtype</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Converts the vcolumns to the input types.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">dtype</div></td> <td><div class="type">dict</div></td> <td><div class = "no">&#10060;</div></td> <td>Dictionary of the different types. Each key of the dictionary must represent a vcolumn. The dictionary must be similar to the following: {"column1": "type1", ... "columnk": "typek"}</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_titanic</span>
<span class="n">titanic</span> <span class="o">=</span> <span class="n">load_titanic</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">titanic</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 14
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">dtypes</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>dtype</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">varchar(100)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"ticket"</b></td><td style="border: 1px solid white;">varchar(36)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"embarked"</b></td><td style="border: 1px solid white;">varchar(20)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"home.dest"</b></td><td style="border: 1px solid white;">varchar(100)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">numeric(10,5)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">varchar(20)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"body"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"age"</b></td><td style="border: 1px solid white;">numeric(6,3)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"name"</b></td><td style="border: 1px solid white;">varchar(164)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"cabin"</b></td><td style="border: 1px solid white;">varchar(30)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">int</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[2]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># doing some data types conversion</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">astype</span><span class="p">({</span><span class="s2">&quot;fare&quot;</span><span class="p">:</span> <span class="s2">&quot;int&quot;</span><span class="p">,</span> <span class="s2">&quot;cabin&quot;</span><span class="p">:</span> <span class="s2">&quot;varchar(1)&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">152</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">152</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">152</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">50</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 14</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">dtypes</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>dtype</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">varchar(100)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"ticket"</b></td><td style="border: 1px solid white;">varchar(36)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"embarked"</b></td><td style="border: 1px solid white;">varchar(20)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"home.dest"</b></td><td style="border: 1px solid white;">varchar(100)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">varchar(20)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"body"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">int</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"age"</b></td><td style="border: 1px solid white;">numeric(6,3)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"name"</b></td><td style="border: 1px solid white;">varchar(164)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"cabin"</b></td><td style="border: 1px solid white;">varchar(1)</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">int</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>