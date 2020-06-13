<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="unsupervised.predict">unsupervised.predict<a class="anchor-link" href="#unsupervised.predict">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">unsupervised</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">vdf</span><span class="p">,</span> 
                     <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Adds the prediction in a vDataFrame.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">vdf</div></td> <td><div class="type">vDataFrame</div></td> <td><div class = "no">&#10060;</div></td> <td>Object used to insert the prediction as a vcolumn.</td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Name of the added vcolumn. If empty, a name will be generated.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : the input object.</p>

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
<div class="prompt input_prompt">In&nbsp;[15]:</div>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalWidthCm</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.10</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">2.90</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.20</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">2.30</td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.30</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
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
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.cluster</span> <span class="k">import</span> <span class="n">KMeans</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">KMeans</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.KMeans_iris&quot;</span><span class="p">,</span>
               <span class="n">n_cluster</span> <span class="o">=</span> <span class="mi">3</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">])</span>
<span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">iris</span><span class="p">,</span> <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;cluster&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cluster</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.10</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">2.90</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.20</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">2.30</td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.30</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[18]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: iris, Number of rows: 150, Number of columns: 6</pre>
</div>

</div>

</div>
</div>

</div>