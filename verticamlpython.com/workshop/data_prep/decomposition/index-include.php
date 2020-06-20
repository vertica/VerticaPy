<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Decomposition">Decomposition<a class="anchor-link" href="#Decomposition">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Decomposition is the process which uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.</p>
<p>Some algorithms are sensible to correlated predictors, the PCA (Principal Component Analysis: Decomposition Technique) can be a solution before applying the algorithm. Besides, some algorithms are sensible to the number of predictors. Only keeping valuable information is a way to solve this problem.</p>
<p>To see how to decompose the data in Vertica ML Python, we will use the well-known 'Iris' dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
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
<p>We can notice that all the predictors are well correlated one to another.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%</span><span class="k">matplotlib</span> inline
<span class="n">vdf</span><span class="o">.</span><span class="n">corr</span><span class="p">()</span>
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

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's compute the PCA of the different elements.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.decomposition</span> <span class="k">import</span> <span class="n">PCA</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="n">drop_model</span>
<span class="n">drop_model</span><span class="p">(</span><span class="s2">&quot;pca_iris&quot;</span><span class="p">)</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">PCA</span><span class="p">(</span><span class="s2">&quot;pca_iris&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> 
                   <span class="s2">&quot;SepalWidthCm&quot;</span><span class="p">,</span>
                   <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">,</span>
                   <span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The model pca_iris was successfully dropped.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[3]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>

=======
columns
=======
index|    name     |  mean  |   sd   
-----+-------------+--------+--------
  1  |petallengthcm| 3.75867| 1.76442
  2  |sepalwidthcm | 3.05400| 0.43359
  3  |sepallengthcm| 5.84333| 0.82807
  4  |petalwidthcm | 1.19867| 0.76316


===============
singular_values
===============
index| value  |explained_variance|accumulated_explained_variance
-----+--------+------------------+------------------------------
  1  | 2.05544|      0.92462     |            0.92462           
  2  | 0.49218|      0.05302     |            0.97763           
  3  | 0.28022|      0.01719     |            0.99482           
  4  | 0.15389|      0.00518     |            1.00000           


====================
principal_components
====================
index|  PC1   |  PC2   |  PC3   |  PC4   
-----+--------+--------+--------+--------
  1  | 0.85657|-0.17577| 0.07252|-0.47972
  2  |-0.08227| 0.72971| 0.59642|-0.32409
  3  | 0.36159| 0.65654|-0.58100| 0.31725
  4  | 0.35884|-0.07471| 0.54906| 0.75112


========
counters
========
   counter_name   |counter_value
------------------+-------------
accepted_row_count|     150     
rejected_row_count|      0      
 iteration_count  |      1      


===========
call_string
===========
SELECT PCA(&#39;public.pca_iris&#39;, &#39;iris&#39;, &#39;&#34;PetalLengthCm&#34;, &#34;SepalWidthCm&#34;, &#34;SepalLengthCm&#34;, &#34;PetalWidthCm&#34;&#39;
USING PARAMETERS scale=false);</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's compute the correlation matrix of the result of the PCA.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">()</span><span class="o">.</span><span class="n">corr</span><span class="p">()</span>
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

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can notice that the predictores are now independant and combined together they have the exact same amount of information than the previous variables. Let's look at the accumulated explained variance of the PCA components.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">explained_variance</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>value</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>explained_variance</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>accumulated_explained_variance</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">2.05544174529956</td><td style="border: 1px solid white;">0.924616207174268</td><td style="border: 1px solid white;">0.924616207174268</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0.492182457659265</td><td style="border: 1px solid white;">0.0530155678505349</td><td style="border: 1px solid white;">0.977631775024803</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0.28022117709794</td><td style="border: 1px solid white;">0.0171851395250069</td><td style="border: 1px solid white;">0.99481691454981</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0.153892907978251</td><td style="border: 1px solid white;">0.00518308545019004</td><td style="border: 1px solid white;">1.0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




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
<p>Most of the information are on the 2 first components with more than 97.7% of explained variance. We can export this result to a vDataFrame.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">n_components</span> <span class="o">=</span> <span class="mi">2</span><span class="p">,</span>
             <span class="n">key_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> 
                            <span class="s2">&quot;SepalWidthCm&quot;</span><span class="p">,</span>
                            <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">,</span>
                            <span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col1</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col2</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1.1</td><td style="border: 1px solid white;">3.0</td><td style="border: 1px solid white;">4.3</td><td style="border: 1px solid white;">0.1</td><td style="border: 1px solid white;">-3.22520044627498</td><td style="border: 1px solid white;">-0.503279909485424</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1.4</td><td style="border: 1px solid white;">2.9</td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">-2.88795856533563</td><td style="border: 1px solid white;">-0.57079802633159</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">3.0</td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">-2.98184266485391</td><td style="border: 1px solid white;">-0.480250048856075</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">3.2</td><td style="border: 1px solid white;">4.4</td><td style="border: 1px solid white;">0.2</td><td style="border: 1px solid white;">-2.99829644283235</td><td style="border: 1px solid white;">-0.334307574590775</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">1.3</td><td style="border: 1px solid white;">2.3</td><td style="border: 1px solid white;">4.5</td><td style="border: 1px solid white;">0.3</td><td style="border: 1px solid white;">-2.85221108156639</td><td style="border: 1px solid white;">-0.932865367469543</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: pca_iris, Number of rows: 150, Number of columns: 6</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can use joins to combine this information with our original dataset.</p>
<p>We now know multiple techniques to prepare our dataset. During the entire Data Preparation cycle, most of the time we need to create new features. That's what we call 'Features Engineering'. It is the main topic of our next lesson.</p>

</div>
</div>
</div>