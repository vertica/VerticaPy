<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.testdw">vDataFrame.testdw<a class="anchor-link" href="#vDataFrame.testdw">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">testdw</span><span class="p">(</span><span class="n">ts</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                  <span class="n">response</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
                  <span class="n">predictors</span><span class="p">:</span> <span class="nb">list</span><span class="p">,</span> 
                  <span class="n">by</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                  <span class="n">print_info</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Durbin Watson test (residuals autocorrelation).</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">ts</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>vcolumn used as timeline. It will be used to order the data. It can be a numerical or type date like (date, datetime, timestamp...) vcolumn.</td> </tr>
    <tr> <td><div class="param_name">response</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Input vcolumn used as response.</td> </tr>
    <tr> <td><div class="param_name">predictors</div></td> <td><div class="type">list</div></td> <td><div class = "no">&#10060;</div></td> <td>Input vcolumns used as predictors.</td> </tr>
    <tr> <td><div class="param_name">by</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>vcolumns used in the partition.</td> </tr>
    <tr> <td><div class="param_name">print_info</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, displays all the test information.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>tuple</b> : <br> [0] result of the test : True if H0 was rejected, <br>
               [1] d : Durbin Watson index (a float between 0 and 4)</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[66]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Let&#39;s look at the trend of the end of 2000 for the number of forest fires in Brazil</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_amazon</span>
<span class="n">amazon</span> <span class="o">=</span> <span class="n">load_amazon</span><span class="p">()</span>
<span class="n">amazon</span><span class="o">.</span><span class="n">analytic</span><span class="p">(</span><span class="n">func</span> <span class="o">=</span> <span class="s2">&quot;lag&quot;</span><span class="p">,</span>
                <span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;number&quot;</span><span class="p">,</span>
                <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;state&quot;</span><span class="p">],</span>
                <span class="n">order_by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;date&quot;</span><span class="p">],</span>
                <span class="n">offset</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span>
                <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;lag_number&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>date</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>number</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>state</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>lag_number</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">1998-01-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">None</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">1998-02-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">1998-03-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">1998-04-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1998-05-01</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Acre</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[66]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: amazon, Number of rows: 6454, Number of columns: 4</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[67]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">amazon</span><span class="o">.</span><span class="n">testdw</span><span class="p">(</span><span class="n">ts</span> <span class="o">=</span> <span class="s2">&quot;date&quot;</span><span class="p">,</span>
              <span class="n">response</span> <span class="o">=</span> <span class="s2">&quot;number&quot;</span><span class="p">,</span> 
              <span class="n">predictors</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;lag_number&quot;</span><span class="p">],</span> 
              <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;state&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>We test the following hypothesis:
(H0) The residuals from regression &#34;number&#34; = LinearRegression(&#34;lag_number&#34;) are stationary
(H1) There is a First Order Auto Correlation in residuals from regression &#34;number&#34; = LinearRegression(&#34;lag_number&#34;)
👍 - The residuals might be stationary
d = 2.13353126698345
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[67]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(True, 2.13353126698345)</pre>
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
    <tr><td><a href="../testdf">vDataFrame.testdf</a></td> <td>Dickey Fuller test.</td></tr>
    <tr><td><a href="../testjb">vDataFrame.testjb</a></td> <td>Jarque Bera test.</td></tr>
    <tr><td><a href="../testmk">vDataFrame.testmk</a></td> <td>Mann Kendall test.</td></tr>
</table>
</div>
</div>
</div>