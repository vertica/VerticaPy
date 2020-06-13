<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.append">vDataFrame.append<a class="anchor-link" href="#vDataFrame.append">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">input_relation</span><span class="p">,</span>
                  <span class="n">expr1</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                  <span class="n">expr2</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                  <span class="n">union_all</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Merges the vDataFrame with another one or an input relation and returns a new vDataFrame.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str / vDataFrame</div></td> <td><div class = "no">&#10060;</div></td> <td>Relation used to do the merging.</td> </tr>
    <tr> <td><div class="param_name">expr1</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of expressions from the current vDataFrame to use during the merging. It must be pure SQL. For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and 'POWER("column", 2)' will work. If empty all the vDataFrame vcolumns will be used. It is highly recommended to write aliases to avoid auto-naming.</td> </tr>
    <tr> <td><div class="param_name">expr2</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of expressions from the input relation to use during the merging. It must be pure SQL. For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and 'POWER("column", 2)' will work. If empty all the input relation columns will be used. It is highly recommended to write aliases to avoid auto-naming.</td> </tr>
    <tr> <td><div class="param_name">union_all</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, the vDataFrame will be merged with the input relation using an 'UNION ALL' instead of an 'UNION'.</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_iris</span>
<span class="n">iris</span> <span class="o">=</span> <span class="n">load_iris</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">iris</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalWidthCm</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.10</td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">3.00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">2.90</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.20</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.30</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">2.30</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: iris, Number of rows: 150, Number of columns: 5
</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Adding another vDataFrame with the same columns name</span>
<span class="n">iris</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">iris</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalWidthCm</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.10</td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">3.00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">2.90</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.20</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.30</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">2.30</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: iris, Number of rows: 300, Number of columns: 5</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Adding another relation with the same columns name</span>
<span class="n">iris</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalWidthCm</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.10</td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">3.00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">2.90</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.20</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.30</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">2.30</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: iris, Number of rows: 300, Number of columns: 5</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Considering only specific columns</span>
<span class="n">iris</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">iris</span><span class="p">,</span>
            <span class="n">expr1</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;SepalLengthCm AS sl&quot;</span><span class="p">,</span> <span class="s2">&quot;PetalLengthCm AS pl&quot;</span><span class="p">],</span>
            <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;SepalLengthCm AS sl&quot;</span><span class="p">,</span> <span class="s2">&quot;PetalLengthCm AS pl&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sl</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pl</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">1.10</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">1.40</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">1.30</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">1.30</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">1.30</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: iris, Number of rows: 300, Number of columns: 2</pre>
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
    <tr><td><a href="../groupby/index.php">vDataFrame.groupby</a></td> <td>Aggregates the vDataFrame.</td></tr>
    <tr><td><a href="../join/index.php">vDataFrame.join</a></td> <td>Joins the vDataFrame with another relation.</td></tr>
    <tr><td><a href="../sort/index.php">vDataFrame.sort</a></td> <td>Sorts the vDataFrame.</td></tr>
</table>
</div>
</div>
</div>