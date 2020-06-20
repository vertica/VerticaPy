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
<p>This example will show you how to use the different methods of a Decomposition Model. We will use the Iris dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[107]:</div>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Species</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalWidthCm</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.10</td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">3.00</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">2.90</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.00</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">3.20</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">Iris-setosa</td><td style="border: 1px solid white;">0.30</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">2.30</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
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
<p>Let's create a PCA model of the different flowers.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[109]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.decomposition</span> <span class="k">import</span> <span class="n">PCA</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">PCA</span><span class="p">(</span><span class="s2">&quot;public.PCA_iris&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalWidthCm&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[109]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>

=======
columns
=======
index|    name     |  mean  |   sd   
-----+-------------+--------+--------
  1  |petalwidthcm | 1.19867| 0.76316
  2  |petallengthcm| 3.75867| 1.76442
  3  |sepallengthcm| 5.84333| 0.82807
  4  |sepalwidthcm | 3.05400| 0.43359


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
  1  | 0.35884|-0.07471| 0.54906| 0.75112
  2  | 0.85657|-0.17577| 0.07252|-0.47972
  3  | 0.36159| 0.65654|-0.58100| 0.31725
  4  |-0.08227| 0.72971| 0.59642|-0.32409


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
SELECT PCA(&#39;public.PCA_iris&#39;, &#39;public.iris&#39;, &#39;&#34;PetalWidthCm&#34;, &#34;PetalLengthCm&#34;, &#34;SepalLengthCm&#34;, &#34;SepalWidthCm&#34;&#39;
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
<p>By fitting the model, new model's attributes will be created. These attributes will be used to simplify the methods usage.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[110]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">X</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[110]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>[&#39;&#34;PetalWidthCm&#34;&#39;, &#39;&#34;PetalLengthCm&#34;&#39;, &#39;&#34;SepalLengthCm&#34;&#39;, &#39;&#34;SepalWidthCm&#34;&#39;]</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[111]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">input_relation</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[111]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&#39;public.iris&#39;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>These attributes will be used when invoking the different model abstractions. The model could also have other useful attributes. In the case of PCA, the 'components', 'explained_variance' and 'mean' attributes can give you useful information about the model.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[112]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">components</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PC1</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PC2</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PC3</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PC4</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0.358843926248216</td><td style="border: 1px solid white;">-0.0747064701350342</td><td style="border: 1px solid white;">0.549060910726603</td><td style="border: 1px solid white;">0.751120560380823</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0.856572105290528</td><td style="border: 1px solid white;">-0.175767403428654</td><td style="border: 1px solid white;">0.0725240754869635</td><td style="border: 1px solid white;">-0.47971898732994</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0.36158967738145</td><td style="border: 1px solid white;">0.656539883285831</td><td style="border: 1px solid white;">-0.580997279827618</td><td style="border: 1px solid white;">0.31725454716854</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">-0.0822688898922142</td><td style="border: 1px solid white;">0.729712371326497</td><td style="border: 1px solid white;">0.596418087938103</td><td style="border: 1px solid white;">-0.324094352417966</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[112]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[113]:</div>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>value</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>explained_variance</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>accumulated_explained_variance</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">2.05544174529956</td><td style="border: 1px solid white;">0.924616207174268</td><td style="border: 1px solid white;">0.924616207174268</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0.492182457659266</td><td style="border: 1px solid white;">0.0530155678505351</td><td style="border: 1px solid white;">0.977631775024803</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0.280221177097939</td><td style="border: 1px solid white;">0.0171851395250068</td><td style="border: 1px solid white;">0.99481691454981</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0.153892907978245</td><td style="border: 1px solid white;">0.00518308545018961</td><td style="border: 1px solid white;">0.999999999999999</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[113]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[114]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">mean</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>mean</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sd</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">petalwidthcm</td><td style="border: 1px solid white;">1.19866666666667</td><td style="border: 1px solid white;">0.763160741700841</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">petallengthcm</td><td style="border: 1px solid white;">3.75866666666667</td><td style="border: 1px solid white;">1.76442041995226</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">sepallengthcm</td><td style="border: 1px solid white;">5.84333333333333</td><td style="border: 1px solid white;">0.828066127977863</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">sepalwidthcm</td><td style="border: 1px solid white;">3.054</td><td style="border: 1px solid white;">0.433594311362174</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[114]:</div>




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
<p>Looking at the SQL code can help you understand how Vertica works.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[115]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">deploySQL</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>APPLY_PCA(&#34;PetalWidthCm&#34;, &#34;PetalLengthCm&#34;, &#34;SepalLengthCm&#34;, &#34;SepalWidthCm&#34; USING PARAMETERS model_name = &#39;public.PCA_iris&#39;, match_by_pos = &#39;true&#39;, cutoff = 1)
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
<p>It is also possible to deploy the inverse PCA.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[116]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">deployInverseSQL</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>APPLY_INVERSE_PCA(&#34;PetalWidthCm&#34;, &#34;PetalLengthCm&#34;, &#34;SepalLengthCm&#34;, &#34;SepalWidthCm&#34; USING PARAMETERS model_name = &#39;public.PCA_iris&#39;, match_by_pos = &#39;true&#39;)
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
<p>It is also possible to use the 'to_vdf' method to get the model vDataFrame. You can choose the number of components to keep.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[117]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">n_components</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col1</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col2</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">-3.22520044627498</td><td style="border: 1px solid white;">-0.503279909485424</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">-2.88795856533563</td><td style="border: 1px solid white;">-0.57079802633159</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">-2.98184266485391</td><td style="border: 1px solid white;">-0.480250048856075</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">-2.99829644283235</td><td style="border: 1px solid white;">-0.334307574590776</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">-2.85221108156639</td><td style="border: 1px solid white;">-0.932865367469544</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[117]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: pca_publiciris, Number of rows: 150, Number of columns: 2</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Or the minimal cumulative explained variance.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[118]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">cutoff</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col1</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">-3.22520044627498</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">-2.88795856533563</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">-2.98184266485391</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">-2.99829644283235</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">-2.85221108156639</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[118]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: col1, Number of rows: 150, dtype: float</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>It is also possible to keep key columns to be able to join the result to the main relation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[119]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="n">cutoff</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> 
             <span class="n">key_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">,</span> 
                            <span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> 
                            <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">,</span> 
                            <span class="s2">&quot;SepalWidthCm&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PetalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalLengthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SepalWidthCm</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>col1</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0.10</td><td style="border: 1px solid white;">1.10</td><td style="border: 1px solid white;">4.30</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">-3.22520044627498</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.40</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">2.90</td><td style="border: 1px solid white;">-2.88795856533563</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">3.00</td><td style="border: 1px solid white;">-2.98184266485391</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0.20</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.40</td><td style="border: 1px solid white;">3.20</td><td style="border: 1px solid white;">-2.99829644283235</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0.30</td><td style="border: 1px solid white;">1.30</td><td style="border: 1px solid white;">4.50</td><td style="border: 1px solid white;">2.30</td><td style="border: 1px solid white;">-2.85221108156639</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[119]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: pca_publiciris, Number of rows: 150, Number of columns: 5</pre>
</div>

</div>

</div>
</div>

</div>