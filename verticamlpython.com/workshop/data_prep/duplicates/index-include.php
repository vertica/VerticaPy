<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Duplicates">Duplicates<a class="anchor-link" href="#Duplicates">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>When merging the different data sources, some duplicates may be created. They are adding a lot of bias to the data. Just imagine running a Telco Marketing campaign and targeting the same person multiple times. You always need to handle them before doing any types of data preparation to avoid unexpected result. Let's use the Iris dataset to understand how to handle duplicated values.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[130]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">vdf</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;iris&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">vdf</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalWidthCm</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">4.3</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.1</td><td style="border: 1px solid white;">1.1</td><td style="border: 1px solid white;">3.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">1.4</td><td style="border: 1px solid white;">2.9</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">3.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">3.2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">4.5</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.3</td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">2.3</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
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
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>To find all the duplicates, you can use the 'duplicated' method.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[131]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">duplicated</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>occurrence</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">4.9</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.1</td><td style="border: 1px solid white;">1.5</td><td style="border: 1px solid white;">3.1</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">5.8</td><td style="border: 1px solid white;">Iris-virginica</td><td style="border: 1px solid white;">1.9</td><td style="border: 1px solid white;">5.1</td><td style="border: 1px solid white;">2.7</td><td style="border: 1px solid white;">2</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[131]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: Duplicated Rows (total = 3), Number of rows: 2, Number of columns: 6</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Using this type of data, we will find flowers which have the exact same characteristics. It doesn't mean that they are real duplicates. There is no need to drop them.</p>
<p>However, if we want to drop the duplicates, it is still possible to do it using the 'drop_duplicates' method.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[132]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">drop_duplicates</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>3 element(s) was/were filtered
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalWidthCm</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">4.3</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.1</td><td style="border: 1px solid white;">1.1</td><td style="border: 1px solid white;">3.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">1.4</td><td style="border: 1px solid white;">2.9</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">3.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">3.2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">4.5</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.3</td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">2.3</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[132]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: iris, Number of rows: 147, Number of columns: 5</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Using this method will add an advanced analytical function in the SQL code generation which is quite expensive. You should use this method after aggregating the data to avoid heavy computations during the entire process.</p>
<p>Let's see in the next lesson how to handle outliers.</p>

</div>
</div>
</div>