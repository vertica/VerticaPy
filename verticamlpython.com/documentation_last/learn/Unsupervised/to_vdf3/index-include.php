<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="unsupervised.to_vdf">unsupervised.to_vdf<a class="anchor-link" href="#unsupervised.to_vdf">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">unsupervised</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">n_components</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span><span class="p">,</span>  
                    <span class="n">cutoff</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span> 
                    <span class="n">key_columns</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span> 
                    <span class="n">inverse</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a vDataFrame of the model.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">n_components</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Number of singular value to return. If set to 0, all the components will be deployed.</td> </tr>
    <tr> <td><div class="param_name">cutoff</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Specifies the minimum accumulated explained variance. Components are taken until the accumulated explained variance reaches this value.</td> </tr>
    <tr> <td><div class="param_name">key_columns</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>Predictors used during the algorithm computation which will be deployed with the singular values.</td> </tr>
    <tr> <td><div class="param_name">inverse</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, the inverse model will be deployed.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : model vDataFrame</p>

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
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.decomposition</span> <span class="k">import</span> <span class="n">PCA</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">PCA</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.pca_iris&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalWidthCm&quot;</span><span class="p">])</span>
<span class="c1"># to_vdf Using Number of components</span>
<span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">n_components</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col1</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col2</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">-3.03394159251135</td><td style="border: 1px solid white;">-0.494095968414455</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">-2.71111913342</td><td style="border: 1px solid white;">-0.564512246710519</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">-2.81163192027064</td><td style="border: 1px solid white;">-0.470233515517571</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">-2.82936296054717</td><td style="border: 1px solid white;">-0.321733948244877</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">-2.71055814048682</td><td style="border: 1px solid white;">-0.926061652961914</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[1]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: pca_publiciris, Number of rows: 150, Number of columns: 2</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># to_vdf Using Number of components &amp; Key Columns</span>
<span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">n_components</span> <span class="o">=</span> <span class="mi">2</span><span class="p">,</span>
             <span class="n">key_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col1</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col2</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">-3.03394159251135</td><td style="border: 1px solid white;">-0.494095968414455</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">-2.71111913342</td><td style="border: 1px solid white;">-0.564512246710519</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">-2.81163192027064</td><td style="border: 1px solid white;">-0.470233515517571</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">-2.82936296054717</td><td style="border: 1px solid white;">-0.321733948244877</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">-2.71055814048682</td><td style="border: 1px solid white;">-0.926061652961914</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[2]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: pca_publiciris, Number of rows: 150, Number of columns: 4</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># to_vdf Using Explained Variance Cutoff</span>
<span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">cutoff</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span>
             <span class="n">key_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col1</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">-3.03394159251135</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">-2.71111913342</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">-2.81163192027064</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">-2.82936296054717</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">-2.71055814048682</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[3]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: pca_publiciris, Number of rows: 150, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>