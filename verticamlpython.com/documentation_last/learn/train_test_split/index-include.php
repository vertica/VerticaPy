<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="train_test_split">train_test_split<a class="anchor-link" href="#train_test_split">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">train_test_split</span><span class="p">(</span><span class="n">input_relation</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
                 <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
                 <span class="n">test_size</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.33</span><span class="p">,</span> 
                 <span class="n">schema_writing</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a temporary table and 2 views which can be used to evaluate a model. The table will include all the main relation information with a test column (boolean) which represents if the data belong to the test or train set.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Input Relation.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">test_size</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Proportion of the test set comparint to the training set.</td> </tr>
    <tr> <td><div class="param_name">schema_writing</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Schema used to write the main relation.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>tuple</b> : (name of the train view, name of the test view)</p>

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
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="n">train_test_split</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(&#39;&#34;public&#34;.VERTICA_ML_PYTHON_SPLIT_iris_67_TRAIN&#39;,
 &#39;&#34;public&#34;.VERTICA_ML_PYTHON_SPLIT_iris_33_TEST&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataFrame</span>
<span class="n">vDataFrame</span><span class="p">(</span><span class="s1">&#39;&quot;public&quot;.VERTICA_ML_PYTHON_SPLIT_iris_67_TRAIN&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>test</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalWidthCm</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.10</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">2.90</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.20</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">2.30</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.30</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[11]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: VERTICA_ML_PYTHON_SPLIT_iris_67_TRAIN, Number of rows: 110, Number of columns: 6</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataFrame</span>
<span class="n">vDataFrame</span><span class="p">(</span><span class="s1">&#39;&quot;public&quot;.VERTICA_ML_PYTHON_SPLIT_iris_33_TEST&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>test</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalWidthCm</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">4.60</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.30</td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">3.40</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">4.70</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.60</td><td style="border: 1px solid white;">3.20</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">4.80</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.60</td><td style="border: 1px solid white;">3.10</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">4.90</td><td style="border: 1px solid white;">Iris-versicolor</td><td style="border: 1px solid white;">1.00</td><td style="border: 1px solid white;">3.30</td><td style="border: 1px solid white;">2.40</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">4.90</td><td style="border: 1px solid white;">Iris-virginica</td><td style="border: 1px solid white;">1.70</td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">2.50</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[12]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: VERTICA_ML_PYTHON_SPLIT_iris_33_TEST, Number of rows: 40, Number of columns: 6</pre>
</div>

</div>

</div>
</div>

</div>