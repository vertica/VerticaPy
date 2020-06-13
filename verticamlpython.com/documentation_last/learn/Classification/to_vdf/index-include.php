<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="classifier.to_vdf">classifier.to_vdf<a class="anchor-link" href="#classifier.to_vdf">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classifier</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">cutoff</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> 
                  <span class="n">all_classes</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Returns the vDataFrame of the Prediction. This method can be used when a model has not the 'predict' method.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">cutoff</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Cutoff used in case of binary classification. It is the probability to accept the category 1.</td> </tr>
    <tr> <td><div class="param_name">all_classes</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, all the classes probabilities will be generated (one column per category).</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : the vDataFrame of the prediction</p>

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
<div class="prompt input_prompt">In&nbsp;[56]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.neighbors</span> <span class="k">import</span> <span class="n">KNeighborsClassifier</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">()</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> 
          <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalWidthCm&quot;</span><span class="p">],</span> 
          <span class="s2">&quot;Species&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">1.00</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">3.60</td><td style="border: 1px solid white;">Iris-setosa</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">0.10</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">Iris-setosa</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">1.20</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">3.20</td><td style="border: 1px solid white;">Iris-setosa</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">1.20</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">Iris-setosa</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">Iris-setosa</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[56]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: knn, Number of rows: 142, Number of columns: 4</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">all_classes</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>SepalWidthCm</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species_Iris-setosa</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species_Iris-versicolor</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Species_Iris-virginica</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">1.00</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">3.60</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">0.10</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">1.20</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">3.20</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">1.20</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">4.00</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[55]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: knn, Number of rows: 142, Number of columns: 6</pre>
</div>

</div>

</div>
</div>

</div>